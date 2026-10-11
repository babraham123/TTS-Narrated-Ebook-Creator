"""Command line entry point: EPUB in, EPUB 3 with Media Overlays out."""

import argparse
import os
import subprocess
import sys

import numpy as np
import soundfile as sf
from bs4 import BeautifulSoup
from tqdm import tqdm

from calliope.epub import (
    create_smil_content,
    detect_styles_dir,
    ensure_sentence_tokenizer,
    find_opf,
    get_audio_duration,
    process_xhtml_inplace,
    unzip_epub,
    update_opf,
    write_nuclear_css,
    zip_dir,
)
from calliope.tts import MODELS, ChatterboxBackend, FakeBackend, resolve_model, synthesize_sentence


def generate_media(basename, segments, backend, audio_dir, smil_dir, rel_text_path):
    mp3_path = os.path.join(audio_dir, f"{basename}.mp3")
    smil_path = os.path.join(smil_dir, f"{basename}.smil")
    
    if os.path.exists(mp3_path) and os.path.exists(smil_path):
        print(f"  [SKIP] Audio exists: {basename}.mp3")
        return get_audio_duration(mp3_path)

    full_audio = []
    sync_data = []
    curr_time = 0.0
    sr = backend.sr

    for seg in tqdm(segments, desc="TTS", leave=False):
        # seg['text'] is clean text
        text = seg['text'].replace("\n", " ").strip()
        if not text: continue

        final_sentence_wav = synthesize_sentence(backend, text, seed=0)

        dur = len(final_sentence_wav) / sr
        start = curr_time
        end = curr_time + dur

        sync_data.append({"id": seg['id'], "start": start, "end": end})
        full_audio.append(final_sentence_wav)
        full_audio.append(np.zeros(int(0.15 * sr), dtype=np.float32))
        curr_time = end + 0.15

    if full_audio:
        wav_path = os.path.join(audio_dir, f"{basename}.wav")
        final_wav = np.concatenate(full_audio)
        sf.write(wav_path, final_wav, sr)

        subprocess.run(["ffmpeg", "-i", wav_path, "-b:a", "128k", "-y", mp3_path, "-loglevel", "error"], check=True)
        os.remove(wav_path)
        total_duration = curr_time
    else:
        total_duration = 0

    smil_content = create_smil_content(basename, sync_data, f"{basename}.mp3", rel_text_path)
    with open(smil_path, "w", encoding="utf-8") as f:
        f.write(smil_content)

    return total_duration


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        prog="calliope",
        description="Generative Audiobook Creator (Chatterbox): Convert EPUB to EPUB 3 with Media Overlays."
    )

    parser.add_argument("input_epub", help="Path to the input .epub file")
    parser.add_argument("--voice", "-v", required=True, help="Path to the reference voice sample (.wav)")
    parser.add_argument("--output", "-o", help="Path for the output .epub file")
    parser.add_argument("--language", "-l", default="en", help="Language code (default: en)")
    parser.add_argument("--model", choices=MODELS, default=None,
                        help="Chatterbox model: turbo (English only) or multilingual. "
                             "Default: turbo for --language en, multilingual otherwise")
    parser.add_argument("--device", default="auto", help="Torch device: auto (default), cuda, cpu")
    parser.add_argument("--skip-audio", action="store_true",
                        help="Debug layout without TTS: narrate with a placeholder tone, no model load")

    args = parser.parse_args(argv)
    try:
        args.model = resolve_model(args.model, args.language)
    except ValueError as e:
        parser.error(str(e))
    return args


def main(argv=None):
    args = parse_arguments(argv)

    INPUT_EPUB = args.input_epub
    SPEAKER_WAV = args.voice
    BOOK_LANGUAGE = args.language

    input_basename = os.path.splitext(os.path.basename(INPUT_EPUB))[0]
    default_filename = f"{input_basename}_Narrated_Chatterbox.epub"

    if args.output:
        if os.path.isdir(args.output) or args.output.endswith(os.sep):
            os.makedirs(args.output, exist_ok=True)
            OUTPUT_EPUB = os.path.join(args.output, default_filename)
        else:
            OUTPUT_EPUB = args.output
            parent_dir = os.path.dirname(os.path.abspath(OUTPUT_EPUB))
            if parent_dir: os.makedirs(parent_dir, exist_ok=True)
    else:
        input_dir = os.path.dirname(os.path.abspath(INPUT_EPUB))
        OUTPUT_EPUB = os.path.join(input_dir, default_filename)

    input_dir = os.path.dirname(os.path.abspath(INPUT_EPUB))
    book_basename = os.path.splitext(os.path.basename(INPUT_EPUB))[0]
    WORK_DIR = os.path.join(input_dir, f".{book_basename}_workdir")

    print(f"=== STARTING CHATTERBOX PIPELINE ===")
    print(f"Input:  {INPUT_EPUB}")
    print(f"Output: {OUTPUT_EPUB}")
    print(f"Features: Smart Split + Ellipsis + Crossfade ENABLED")

    if not os.path.exists(INPUT_EPUB):
        print(f"Error: Input file not found: {INPUT_EPUB}")
        return 1
    if not os.path.exists(SPEAKER_WAV):
        print(f"Error: Voice sample not found: {SPEAKER_WAV}")
        return 1

    ensure_sentence_tokenizer()

    unzip_epub(INPUT_EPUB, WORK_DIR)
    opf_path, oebps_dir = find_opf(WORK_DIR)

    with open(opf_path, 'r') as f: soup = BeautifulSoup(f, 'xml')
    styles_dir, styles_rel_prefix = detect_styles_dir(soup.manifest, oebps_dir)

    audio_dir = os.path.join(oebps_dir, "Audio")
    smil_dir = os.path.join(oebps_dir, "Smil")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(smil_dir, exist_ok=True)
    os.makedirs(styles_dir, exist_ok=True)

    css_name = "sync_highlight.css"
    write_nuclear_css(styles_dir, css_name)

    print("\n--- Processing Content ---")
    manifest = soup.manifest
    spine = soup.spine

    if args.skip_audio:
        tts = FakeBackend()
        print("[INFO] --skip-audio: narrating with a placeholder tone")
    else:
        print(f"Loading Chatterbox {args.model} on {args.device}...")
        tts = ChatterboxBackend(args.model, BOOK_LANGUAGE, SPEAKER_WAV, args.device)

    manifest_additions = []
    manifest_additions.append({"id": "sync_css", "href": f"{styles_rel_prefix}/{css_name}", "media-type": "text/css"})

    spine_mapping = {}
    total_durations = {}
    global_id = 1
    processed_files = set()

    for itemref in spine.find_all("itemref"):
        item_id = itemref['idref']
        item = manifest.find("item", id=item_id)
        if not item or item['media-type'] != "application/xhtml+xml":
            continue

        filename = item['href']
        full_path = os.path.join(oebps_dir, filename)
        basename = os.path.splitext(os.path.basename(filename))[0]

        if full_path in processed_files or item.get('properties') == 'nav':
            continue
        processed_files.add(full_path)

        rel_text_path = f"../{filename}"
        rel_css = os.path.relpath(os.path.join(styles_dir, css_name), os.path.dirname(full_path))

        print(f"Processing: {basename}")

        segments, global_id = process_xhtml_inplace(full_path, global_id, rel_css, BOOK_LANGUAGE)

        if not segments: continue

        duration = generate_media(basename, segments, tts, audio_dir, smil_dir, rel_text_path)

        smil_id = f"smil_{basename}"
        audio_id = f"audio_{basename}"

        manifest_additions.append({"id": smil_id, "href": f"Smil/{basename}.smil", "media-type": "application/smil+xml"})
        manifest_additions.append({"id": audio_id, "href": f"Audio/{basename}.mp3", "media-type": "audio/mpeg"})

        spine_mapping[item_id] = smil_id
        total_durations[smil_id] = duration

    print("\n--- Updating Metadata ---")
    update_opf(opf_path, manifest_additions, spine_mapping, total_durations)

    print("\n--- Packaging EPUB ---")
    zip_dir(WORK_DIR, OUTPUT_EPUB)

    print(f"\nSUCCESS! Saved to: {OUTPUT_EPUB}")
    return 0
