"""The narration pipeline: EPUB in, EPUB 3 with Media Overlays out, resumable via a work dir.

Phases:
  1. extract the input into <work>/build and wrap every sentence in a <span id>;
  2. make sure every sentence has an accepted clip in the cache, synthesizing only what is
     missing (the TTS model is loaded only if something is missing). Each clip is checked
     (calliope.qa) and retried with a new seed up to max_retries times; if no attempt passes,
     the best one is kept and flagged;
  3. optionally transcribe clips with Whisper and flag high error rates;
  4. assemble chapter audio, SMIL and OPF from the cached clips and package the EPUB.
Phase 3 always reads clips back from the cache, so a fresh run and a resumed run build the
EPUB from identical audio and produce identical bytes.
"""

import os
import shutil
import time
import urllib.parse
from dataclasses import dataclass, field
from importlib import metadata

import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm

from calliope.cache import ClipCache, attempt_seed, clip_key, file_sha256
from calliope.epub import (
    MP3_BITRATE,
    SENTENCE_GAP_SECONDS,
    assemble_chapter,
    create_smil_content,
    detect_styles_dir,
    encode_mp3,
    ensure_sentence_tokenizer,
    find_opf,
    process_xhtml_inplace,
    unzip_epub,
    update_opf,
    write_nuclear_css,
    zip_dir,
)
from calliope.files import atomic_write_text
from calliope.qa import QAConfig, error_rate, evaluate
from calliope.tts import CHUNK_CHAR_LIMIT, CROSSFADE_MS, synthesize_sentence

CSS_NAME = "sync_highlight.css"


@dataclass
class Options:
    input: str
    output: str
    voice: str
    lang: str
    model: str  # "turbo", "multilingual" or "fake"
    work_dir: str
    seed: int = 0
    max_retries: int = 2
    qa: QAConfig = field(default_factory=QAConfig)
    wer_threshold: float = 0.3
    # Test hook: sentence index -> number of leading attempts replaced by silence (None = all).
    force_silence: dict = field(default_factory=dict)


@dataclass
class Sentence:
    index: int  # position in the book, 0-based, counting only sentences with speakable text
    chapter: str
    text_id: str
    text: str
    key: str = ""


@dataclass
class Chapter:
    item_id: str
    name: str
    href: str
    sentences: list


def model_id(model):
    if model == "fake":
        return "fake"
    try:
        return f"chatterbox-tts=={metadata.version('chatterbox-tts')}:{model}"
    except metadata.PackageNotFoundError:
        return f"chatterbox-tts:{model}"


def _prepare_book(opts, build_dir):
    """Extract the book, wrap sentences and return (opf_path, styles info, chapters)."""
    ensure_sentence_tokenizer()
    unzip_epub(opts.input, build_dir)
    opf_path, oebps_dir = find_opf(build_dir)
    with open(opf_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "xml")

    styles_dir, styles_rel_prefix = detect_styles_dir(soup.manifest, oebps_dir)
    os.makedirs(styles_dir, exist_ok=True)
    write_nuclear_css(styles_dir, CSS_NAME)

    chapters, seen_paths, used_names = [], set(), set()
    global_id, index = 1, 0
    for itemref in soup.spine.find_all("itemref"):
        item = soup.manifest.find("item", id=itemref["idref"])
        if not item or item.get("media-type") != "application/xhtml+xml" or "nav" in item.get("properties", "").split():
            continue
        href = item["href"]
        full_path = os.path.join(oebps_dir, urllib.parse.unquote(href))
        if full_path in seen_paths:
            continue
        seen_paths.add(full_path)

        name = os.path.splitext(os.path.basename(urllib.parse.unquote(href)))[0]
        if name in used_names:
            name = f"{name}_{item['id']}"
        used_names.add(name)

        rel_css = os.path.relpath(os.path.join(styles_dir, CSS_NAME), os.path.dirname(full_path))
        segments, global_id = process_xhtml_inplace(full_path, global_id, rel_css, opts.lang)
        sentences = []
        for seg in segments:
            text = seg["text"].replace("\n", " ").strip()
            if text:
                sentences.append(Sentence(index, name, seg["id"], text))
                index += 1
        if sentences:
            chapters.append(Chapter(item["id"], name, href, sentences))

    return opf_path, oebps_dir, f"{styles_rel_prefix}/{CSS_NAME}" if styles_rel_prefix else CSS_NAME, chapters


def _forced_silent(opts, index, attempt):
    if index not in opts.force_silence:
        return False
    limit = opts.force_silence[index]
    return limit is None or attempt < limit


def _synthesize_clip(backend, s, cache, opts):
    """Run attempts for one sentence until one passes QA or the retry budget is spent, continuing
    after any attempts already cached. Stores the best attempt. Returns (QAResult, total
    attempts, attempts made now)."""
    best = None  # (badness, attempt, flac bytes, result)
    first = 0
    cached = cache.load(s.key)
    if cached is not None:
        wav, sr, meta = cached
        result = evaluate(wav, sr, s.text, opts.qa)
        best = (result.badness, meta["best_attempt"], cache.encode(wav, sr)[0], result)
        first = meta["attempts"]

    last_error = None
    attempt = first
    for attempt in range(first, opts.max_retries + 1):
        try:
            wav = synthesize_sentence(backend, s.text, attempt_seed(s.key, attempt))
        except Exception as e:  # a failed attempt is retried like a rejected one
            last_error = e
            continue
        if _forced_silent(opts, s.index, attempt):
            wav = np.zeros_like(wav)
        data, stored_wav = cache.encode(wav, backend.sr)
        result = evaluate(stored_wav, backend.sr, s.text, opts.qa)
        if best is None or result.badness < best[0]:
            best = (result.badness, attempt, data, result)
        if result.passed:
            break
    if best is None:
        raise RuntimeError(f"synthesis failed for sentence {s.index} ({s.text_id}): {last_error}")

    attempts = attempt + 1
    _badness, best_attempt, data, result = best
    cache.store(s.key, data, {"text": s.text, "attempts": attempts, "best_attempt": best_attempt})
    return result, attempts, attempts - first


def _transcribe_clips(unique, qa, cache, transcriber_factory, lang):
    """Return key -> (error rate, transcript) for every clip, transcribing uncached ones."""
    transcriber = None  # loaded only if some transcript is not cached
    name = transcriber_factory.name
    out = {}
    for key, s in tqdm(unique.items(), desc="Whisper", unit="sentence"):
        attempts = qa[key][1]
        text = cache.load_transcript(key, name, attempts)
        if text is None:
            if transcriber is None:
                transcriber = transcriber_factory()
            wav, sr, _meta = cache.load(key)
            text = transcriber.transcribe(wav, sr)
            cache.store_transcript(key, name, attempts, text)
        out[key] = (error_rate(s.text, text, lang), text)
    return out


def run(opts, make_backend, transcriber_factory=None, log=print):
    """Narrate opts.input into opts.output.

    make_backend() is called at most once, and only when some clip still needs synthesis.
    transcriber_factory, if given, enables the Whisper check: it must have a `name` attribute
    and return an object with transcribe(wav, sr). Returns run statistics."""
    started = time.monotonic()
    cache = ClipCache(opts.work_dir)
    build_dir = os.path.join(opts.work_dir, "build")

    book_sha = file_sha256(opts.input)
    voice_sha = file_sha256(opts.voice)
    settings = {"seed": opts.seed, "chunk_chars": CHUNK_CHAR_LIMIT, "crossfade_ms": CROSSFADE_MS}
    mid = model_id(opts.model)

    log("--- Processing content ---")
    opf_path, oebps_dir, css_href, chapters = _prepare_book(opts, build_dir)
    sentences = [s for ch in chapters for s in ch.sentences]
    for s in sentences:
        s.key = clip_key(book_sha, s.text, mid, opts.lang, voice_sha, settings)

    unique = {}
    for s in sentences:
        unique.setdefault(s.key, s)

    qa = {}  # key -> (QAResult, attempts)
    todo = []
    for key, s in unique.items():
        cached = cache.load(key)
        if cached is None:
            todo.append(s)
            continue
        wav, sr, meta = cached
        result = evaluate(wav, sr, s.text, opts.qa)
        qa[key] = (result, meta["attempts"])
        if not result.passed and meta["attempts"] < opts.max_retries + 1:
            todo.append(s)  # a higher --max-retries allows more attempts on a flagged clip
    log(f"{len(sentences)} sentences in {len(chapters)} chapters; "
        f"{len(unique) - len(todo)} clips done, {len(todo)} to synthesize")

    synth_seconds, attempts_this_run = 0.0, 0
    if todo:
        backend = make_backend()
        t0 = time.monotonic()
        for s in tqdm(todo, desc="TTS", unit="sentence"):
            result, attempts, made = _synthesize_clip(backend, s, cache, opts)
            qa[s.key] = (result, attempts)
            attempts_this_run += made
        synth_seconds = time.monotonic() - t0

    wer = {}
    if transcriber_factory is not None:
        wer = _transcribe_clips(unique, qa, cache, transcriber_factory, opts.lang)

    flagged = []
    for s in sentences:
        result, attempts = qa[s.key]
        entry_wer = wer.get(s.key)
        high_wer = entry_wer is not None and entry_wer[0] > opts.wer_threshold
        if result.passed and not high_wer:
            continue
        reasons = list(result.reasons) + (["high_wer"] if high_wer else [])
        flagged.append({
            "index": s.index, "chapter": s.chapter, "id": s.text_id, "text": s.text,
            "reasons": reasons, "attempts": attempts, "duration": result.duration,
            "expected_duration": result.expected, "silence_fraction": result.silence_fraction,
            "wer": None if entry_wer is None else round(entry_wer[0], 3),
            "transcript": None if entry_wer is None else entry_wer[1],
        })
    if flagged:
        log(f"{len(flagged)} sentences flagged by quality checks")

    log("--- Assembling audio and overlays ---")
    audio_dir = os.path.join(oebps_dir, "Audio")
    smil_dir = os.path.join(oebps_dir, "Smil")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(smil_dir, exist_ok=True)

    manifest_additions = [{"id": "sync_css", "href": css_href, "media-type": "text/css"}]
    spine_mapping, durations_ms = {}, {}
    sr = None
    for ch in chapters:
        sentence_wavs = []
        for s in ch.sentences:
            wav, sr, _meta = cache.load(s.key)
            sentence_wavs.append((s.text_id, wav))
        audio, clips, duration_ms = assemble_chapter(sentence_wavs, sr)

        encoding = {"format": "mp3", "bitrate": MP3_BITRATE, "sr": sr, "gap": SENTENCE_GAP_SECONDS}
        cached_mp3 = cache.chapter_path([s.key for s in ch.sentences], encoding)
        if not os.path.exists(cached_mp3):
            encode_mp3(audio, sr, cached_mp3)
        shutil.copyfile(cached_mp3, os.path.join(audio_dir, f"{ch.name}.mp3"))

        smil = create_smil_content(f"{ch.name}_seq", clips, f"../Audio/{ch.name}.mp3", f"../{ch.href}")
        atomic_write_text(os.path.join(smil_dir, f"{ch.name}.smil"), smil)

        smil_id = f"smil_{ch.name}"
        manifest_additions.append({"id": smil_id, "href": f"Smil/{ch.name}.smil", "media-type": "application/smil+xml"})
        manifest_additions.append({"id": f"audio_{ch.name}", "href": f"Audio/{ch.name}.mp3", "media-type": "audio/mpeg"})
        spine_mapping[ch.item_id] = smil_id
        durations_ms[smil_id] = duration_ms

    update_opf(opf_path, manifest_additions, spine_mapping, durations_ms)

    log("--- Packaging EPUB ---")
    zip_dir(build_dir, opts.output)
    shutil.rmtree(build_dir, ignore_errors=True)

    return {
        "chapters": len(chapters),
        "sentences": len(sentences),
        "unique_clips": len(unique),
        "synthesized": len(todo),
        "cached": len(unique) - len(todo),
        "attempts_this_run": attempts_this_run,
        "retries": sum(attempts - 1 for _result, attempts in qa.values()),
        "flagged": flagged,
        "audio_seconds": sum(durations_ms.values()) / 1000.0,
        "synthesis_seconds": synth_seconds,
        "wall_seconds": time.monotonic() - started,
    }
