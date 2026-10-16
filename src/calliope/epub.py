"""EPUB handling: extraction, sentence wrapping, SMIL generation, OPF update and packaging."""

import os
import re
import shutil
import subprocess
import unicodedata
import zipfile
from xml.sax.saxutils import escape as xml_escape, quoteattr

import nltk
import numpy as np
from bs4 import BeautifulSoup, CData, NavigableString, Tag

from calliope.files import atomic_output, atomic_write_text
from calliope.normalize import GROUP_MARK, normalize_symbols


# -------------------------------------------------------------------------
# EPUB INFRASTRUCTURE
# -------------------------------------------------------------------------
def unzip_epub(epub_path, extract_to):
    # Always start from a pristine copy: sentence wrapping edits files in place, so reusing an
    # earlier extraction would wrap sentences twice.
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    with zipfile.ZipFile(epub_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def find_opf(work_dir):
    container = os.path.join(work_dir, "META-INF", "container.xml")
    with open(container, 'r') as f:
        soup = BeautifulSoup(f, 'xml')
    rootfile = soup.find('rootfile')
    full_path = os.path.join(work_dir, rootfile['full-path'])
    return full_path, os.path.dirname(full_path)

def detect_styles_dir(manifest_soup, oebps_dir):
    css_item = manifest_soup.find('item', attrs={'media-type': 'text/css'})
    if css_item:
        href = css_item['href']
        style_folder = os.path.dirname(href)
        return os.path.join(oebps_dir, style_folder), style_folder
    else:
        return os.path.join(oebps_dir, "Styles"), "Styles"

def write_nuclear_css(styles_dir, css_filename):
    css_path = os.path.join(styles_dir, css_filename)
    nuclear_css = """
    /* Force High Priority Highlighting */
    span.-epub-media-overlay-active, .-epub-media-overlay-active,
    span.epub-media-overlay-active, .epub-media-overlay-active {
        background-color: #ffff00 !important;
        color: black !important;
        border-radius: 2px;
    }
    @media (prefers-color-scheme: dark) {
        span.-epub-media-overlay-active, .-epub-media-overlay-active,
        span.epub-media-overlay-active, .epub-media-overlay-active {
            background-color: #7b1fa2 !important;
            color: white !important;
        }
    }
    """
    with open(css_path, "w") as f:
        f.write(nuclear_css)

def ensure_sentence_tokenizer():
    # NLTK >= 3.9 loads the pickle-free "punkt_tab" data instead of "punkt".
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

# -------------------------------------------------------------------------
# TEXT PROCESSING (HTML-PRESERVING DOM TRAVERSAL)
# -------------------------------------------------------------------------
_APOSTROPHE = re.compile(r"(?<=\w)['\N{LEFT SINGLE QUOTATION MARK}\N{RIGHT SINGLE QUOTATION MARK}](?=\w)")
_QUOTES = re.compile(
    r"[\"'\N{LEFT DOUBLE QUOTATION MARK}\N{RIGHT DOUBLE QUOTATION MARK}"
    r"\N{LEFT SINGLE QUOTATION MARK}\N{RIGHT SINGLE QUOTATION MARK}"
    r"\N{LEFT-POINTING DOUBLE ANGLE QUOTATION MARK}\N{RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK}]"
)
# Letters and digits of every script survive; CJK sentence punctuation is kept for prosody.
# A no-break or thin space between digit groups ("12\N{NO-BREAK SPACE}345") is a thousands
# separator; NFKC would fold it into an ordinary space.
_DIGIT_GROUP_SPACE = re.compile(r"(?<=\d)[\N{NO-BREAK SPACE}\N{NARROW NO-BREAK SPACE}\N{THIN SPACE}](?=\d{3}(?!\d))")
_DISALLOWED = re.compile(r"[^\w\s.,?!;:'\-\N{IDEOGRAPHIC FULL STOP}\N{IDEOGRAPHIC COMMA}]")
_APOSTROPHE_PLACEHOLDER = chr(0xE000)  # private-use character, never present in book text

# Punkt models shipped in NLTK's punkt_tab, keyed by the TTS language code.
PUNKT_LANGUAGES = {
    "cs": "czech", "da": "danish", "de": "german", "el": "greek", "en": "english",
    "es": "spanish", "et": "estonian", "fi": "finnish", "fr": "french", "it": "italian",
    "ml": "malayalam", "nl": "dutch", "no": "norwegian", "pl": "polish", "pt": "portuguese",
    "ru": "russian", "sl": "slovene", "sv": "swedish", "tr": "turkish",
}
_CJK_SENTENCE_END = re.compile(
    r"(?<=[\N{IDEOGRAPHIC FULL STOP}\N{FULLWIDTH EXCLAMATION MARK}\N{FULLWIDTH QUESTION MARK}!?])"
)

BLOCK_TAGS = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'blockquote', 'div']
VOID_TAGS = {'br', 'img', 'hr', 'area', 'base', 'col', 'embed', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}
XHTML_NS = "http://www.w3.org/1999/xhtml"
OPS_NS = "http://www.idpf.org/2007/ops"


def split_sentences(text, lang="en"):
    """Sentence-split with the punkt model for `lang` (English fallback); punkt has no CJK model."""
    if lang in ("zh", "ja"):
        return [s for s in (part.strip() for part in _CJK_SENTENCE_END.split(text)) if s]
    return nltk.sent_tokenize(text, language=PUNKT_LANGUAGES.get(lang, "english"))


def clean_text_for_tts(text, lang="en"):
    """
    Cleans text for TTS only. Visual text remains untouched.
    """
    text = _DIGIT_GROUP_SPACE.sub(GROUP_MARK, text)
    text = unicodedata.normalize('NFKC', text)
    text = normalize_symbols(text, lang).replace(GROUP_MARK, ' ')
    text = re.sub(r'\.\s\.\s\.', '...', text)
    text = re.sub(r'\s*—\s*', ', ', text).replace('–', '-').replace('_', ' ')
    text = _APOSTROPHE.sub(_APOSTROPHE_PLACEHOLDER, text)
    text = _QUOTES.sub('', text)
    text = text.replace(_APOSTROPHE_PLACEHOLDER, "'")
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = _DISALLOWED.sub("", text)
    return re.sub(r'\s+', ' ', text).strip()


def _qualified_name(node):
    return f"{node.prefix}:{node.name}" if node.prefix else node.name


def _open_tag(name, attrs):
    attr_str = " ".join(f"{k}={quoteattr(v)}" for k, v in attrs.items())
    return f"<{name} {attr_str}>" if attr_str else f"<{name}>"


def _sentence_boundaries(full_text_raw, sentences_clean):
    """Map each whitespace-normalised sentence back to its end offset in the raw text."""
    split_indices = []
    cursor_raw = 0
    for sent_clean in sentences_clean:
        pattern = re.compile(re.escape(sent_clean).replace(r'\ ', r'\s+'))
        match = pattern.search(full_text_raw, cursor_raw)
        if match:
            end_raw = match.end()
            while end_raw < len(full_text_raw) and full_text_raw[end_raw].isspace():
                end_raw += 1
            cursor_raw = end_raw
        else:
            cursor_raw = len(full_text_raw)  # Fallback
        split_indices.append(cursor_raw)
    return split_indices


def process_xhtml_inplace(filepath, global_id_start, css_rel_path, lang="en"):
    """Wrap every sentence of `filepath` in <span id="fNNNNNN"> and return the TTS segments.

    Raises on malformed content rather than silently leaving a chapter un-narrated.
    """
    filename = os.path.basename(filepath)
    print(f"Processing: {filename}")

    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'xml')

    # Cleanup & CSS
    for tag in soup.find_all("a"):
        if "noteref" in tag.get("class", []) or tag.get("role") == "doc-noteref": tag.decompose()

    head = soup.find('head')
    if head:
        css_name = os.path.basename(css_rel_path)
        if not any(css_name in l.get('href', '') for l in head.find_all('link')):
            head.append(soup.new_tag("link", rel="stylesheet", href=css_rel_path, type="text/css"))

    segments = []
    current_id = global_id_start

    for tag in soup.find_all(BLOCK_TAGS):
        if tag.find(BLOCK_TAGS): continue

        # get_text() counts only text and CDATA nodes, so traverse() must skip everything else
        # (comments, processing instructions) to keep the character offsets aligned.
        full_text_raw = tag.get_text()
        if not full_text_raw.strip(): continue

        full_text_clean = re.sub(r'\s+', ' ', full_text_raw).strip()
        sentences_clean = split_sentences(full_text_clean, lang)
        if not sentences_clean: continue

        split_indices = _sentence_boundaries(full_text_raw, sentences_clean)

        parts = []
        current_sent_idx = 0
        current_char_count = 0

        def open_sentence(open_tags):
            nonlocal current_id
            seg_id = f"f{current_id:06d}"
            current_id += 1
            segments.append({"id": seg_id, "text": clean_text_for_tts(sentences_clean[current_sent_idx], lang)})
            parts.append(f'<span id="{seg_id}">')
            for t_name, t_attrs in open_tags:
                parts.append(_open_tag(t_name, t_attrs))

        def close_sentence(open_tags):
            nonlocal current_sent_idx
            for t_name, _ in reversed(open_tags):
                parts.append(f"</{t_name}>")
            parts.append("</span>")
            current_sent_idx += 1
            if current_sent_idx < len(sentences_clean):
                open_sentence(open_tags)
            else:
                # Trailing text outside any sentence still needs its inline wrappers.
                for t_name, t_attrs in open_tags:
                    parts.append(_open_tag(t_name, t_attrs))

        def traverse(node, open_tags):
            nonlocal current_char_count

            if isinstance(node, NavigableString):
                if type(node) not in (NavigableString, CData):
                    return
                text = str(node)
                while text:
                    if current_sent_idx >= len(split_indices):
                        parts.append(xml_escape(text))
                        current_char_count += len(text)
                        break

                    remaining_len = split_indices[current_sent_idx] - current_char_count
                    if remaining_len <= 0:
                        # Boundary reached exactly at a node edge: switch sentence, consume nothing.
                        close_sentence(open_tags)
                        continue

                    if len(text) <= remaining_len:
                        parts.append(xml_escape(text))
                        current_char_count += len(text)
                        break

                    parts.append(xml_escape(text[:remaining_len]))
                    current_char_count += remaining_len
                    text = text[remaining_len:]
                    close_sentence(open_tags)

            elif isinstance(node, Tag):
                name = _qualified_name(node)
                attrs = {k: " ".join(v) if isinstance(v, list) else v for k, v in node.attrs.items()}
                if node.name in VOID_TAGS:
                    parts.append(_open_tag(name, attrs)[:-1] + " />")
                else:
                    open_tags.append((name, attrs))
                    parts.append(_open_tag(name, attrs))
                    for child in node.contents:
                        traverse(child, open_tags)
                    parts.append(f"</{name}>")
                    open_tags.pop()

        open_sentence([])
        for child in list(tag.contents):
            traverse(child, [])
        if current_sent_idx < len(sentences_clean):
            parts.append("</span>")

        # Declare the namespaces the fragment may use so prefixed names (epub:type) parse.
        wrapped_content = f'<body xmlns="{XHTML_NS}" xmlns:epub="{OPS_NS}">{"".join(parts)}</body>'
        new_soup = BeautifulSoup(wrapped_content, 'xml')
        if new_soup.body is None:
            raise ValueError(f"{filename}: could not re-parse sentence markup in <{tag.name}>")
        tag.clear()
        for child in list(new_soup.body.contents):
            tag.append(child)

    with open(filepath, 'w', encoding='utf-8') as f:
        # Use str(soup) instead of prettify to prevent injecting unwanted whitespace
        f.write(str(soup))
    return segments, current_id


# -------------------------------------------------------------------------
# AUDIO ENCODING & SMIL GENERATION
# -------------------------------------------------------------------------
MP3_BITRATE = "64k"


def encode_mp3(wav, sr, mp3_path):
    """Encode mono float audio to MP3 atomically. -bitexact keeps encoder version strings out
    of the file so identical input gives identical bytes."""
    pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()
    with atomic_output(mp3_path) as tmp:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
             "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
             "-map_metadata", "-1", "-c:a", "libmp3lame", "-b:a", MP3_BITRATE,
             "-bitexact", "-f", "mp3", tmp],
            input=pcm, check=True,
        )


SENTENCE_GAP_SECONDS = 0.15


def samples_to_ms(samples, sr):
    return (samples * 1000 + sr // 2) // sr


def assemble_chapter(sentence_wavs, sr, gap_seconds=SENTENCE_GAP_SECONDS):
    """Concatenate (text_id, wav) pairs with a silence gap between sentences.

    Returns (audio, clips, duration_ms). Timings come from integer sample counts, so they are
    exact; each clip runs until the next one begins (the gap is highlighted with the
    preceding sentence) and the last ends with the audio, so the overlay has no holes.
    """
    gap = np.zeros(int(round(gap_seconds * sr)), dtype=np.float32)
    pieces, starts = [], []
    cursor = 0
    for i, (_text_id, wav) in enumerate(sentence_wavs):
        if i:
            pieces.append(gap)
            cursor += len(gap)
        starts.append(cursor)
        pieces.append(np.asarray(wav, dtype=np.float32))
        cursor += len(wav)
    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    bounds = [samples_to_ms(s, sr) for s in starts] + [samples_to_ms(cursor, sr)]
    clips = [(text_id, bounds[i], bounds[i + 1]) for i, (text_id, _) in enumerate(sentence_wavs)]
    return audio, clips, bounds[-1]


def format_clock(ms):
    """SMIL/EPUB full clock value (h:mm:ss.fff) from integer milliseconds."""
    h, rest = divmod(ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, frac = divmod(rest, 1000)
    return f"{h}:{m:02d}:{s:02d}.{frac:03d}"


def create_smil_content(seq_id, clips, audio_href, text_href):
    """clips: [(text_id, begin_ms, end_ms)] in reading order."""
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<smil xmlns="http://www.w3.org/ns/SMIL" xmlns:epub="http://www.idpf.org/2007/ops" version="3.0">',
        '  <body>',
        f'    <seq id="{seq_id}" epub:textref="{text_href}">',
    ]
    for t_id, begin_ms, end_ms in clips:
        lines += [
            f'      <par id="par_{t_id}">',
            f'        <text src="{text_href}#{t_id}"/>',
            f'        <audio src="{audio_href}" clipBegin="{format_clock(begin_ms)}" clipEnd="{format_clock(end_ms)}"/>',
            '      </par>',
        ]
    lines += ['    </seq>', '  </body>', '</smil>', '']
    return "\n".join(lines)


# -------------------------------------------------------------------------
# OPF & PACKAGING
# -------------------------------------------------------------------------
ACTIVE_CLASS = "-epub-media-overlay-active"


def update_opf(opf_path, manifest_additions, spine_mapping, durations_ms):
    """Add overlay items, media-overlay attributes and media:duration / media:active-class meta.

    durations_ms maps SMIL manifest id -> duration; the total is their exact sum so the
    per-overlay and book durations always agree.
    """
    with open(opf_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'xml')

    for meta in soup.metadata.find_all("meta", property="media:duration"): meta.decompose()
    for meta in soup.metadata.find_all("meta", property="media:active-class"): meta.decompose()

    meta_dur = soup.new_tag("meta", property="media:duration")
    meta_dur.string = format_clock(sum(durations_ms.values()))
    soup.metadata.append(meta_dur)

    for smil_id, dur_ms in durations_ms.items():
        meta = soup.new_tag("meta", property="media:duration")
        meta['refines'] = f"#{smil_id}"
        meta.string = format_clock(dur_ms)
        soup.metadata.append(meta)

    meta_active = soup.new_tag("meta", property="media:active-class")
    meta_active.string = ACTIVE_CLASS
    soup.metadata.append(meta_active)

    manifest = soup.manifest
    for item in manifest_additions:
        existing = manifest.find("item", id=item['id'])
        if existing: existing.decompose()

    for item in manifest_additions:
        manifest.append(soup.new_tag("item", attrs=item))

    for text_id, smil_id in spine_mapping.items():
        item = manifest.find("item", id=text_id)
        if item:
            item['media-overlay'] = smil_id

    # str() rather than prettify(): prettify wraps meta values in whitespace.
    atomic_write_text(opf_path, str(soup))


ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_STORED_SUFFIXES = (".mp3", ".m4a", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".woff", ".woff2")


def _zip_info(name, compress_type):
    info = zipfile.ZipInfo(name, date_time=ZIP_EPOCH)
    info.compress_type = compress_type
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def zip_dir(folder, output_path):
    """Package `folder` as an EPUB atomically and reproducibly: mimetype first and stored,
    then every other file in sorted order with fixed timestamps and permissions."""
    rel_paths = []
    for root, _dirs, files in os.walk(folder):
        for file in files:
            rel = os.path.relpath(os.path.join(root, file), folder).replace(os.sep, "/")
            if rel != "mimetype":
                rel_paths.append(rel)
    rel_paths.sort()

    with atomic_output(output_path) as tmp:
        with zipfile.ZipFile(tmp, 'w') as zipf:
            zipf.writestr(_zip_info("mimetype", zipfile.ZIP_STORED), b"application/epub+zip")
            for rel in rel_paths:
                stored = rel.lower().endswith(_STORED_SUFFIXES)
                info = _zip_info(rel, zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
                with open(os.path.join(folder, rel), "rb") as f:
                    zipf.writestr(info, f.read(), compresslevel=None if stored else 9)
