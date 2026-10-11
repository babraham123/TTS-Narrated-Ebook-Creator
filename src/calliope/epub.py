"""EPUB handling: extraction, sentence wrapping, SMIL generation, OPF update and packaging."""

import os
import re
import shutil
import subprocess
import unicodedata
import zipfile
from xml.sax.saxutils import escape as xml_escape, quoteattr

import nltk
from bs4 import BeautifulSoup, CData, NavigableString, Tag


# -------------------------------------------------------------------------
# EPUB INFRASTRUCTURE
# -------------------------------------------------------------------------
def unzip_epub(epub_path, extract_to):
    if os.path.exists(extract_to) and os.path.exists(os.path.join(extract_to, "META-INF")):
        return
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


def clean_text_for_tts(text):
    """
    Cleans text for TTS only. Visual text remains untouched.
    """
    text = unicodedata.normalize('NFKC', text)
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
            segments.append({"id": seg_id, "text": clean_text_for_tts(sentences_clean[current_sent_idx])})
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
# SMIL GENERATION
# -------------------------------------------------------------------------
def get_audio_duration(filepath):
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", filepath]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())
    except: return 0.0

def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h)}:{int(m):02d}:{s:06.3f}"

def create_smil_content(basename, segments, audio_filename, rel_text_path):
    xml = '<?xml version="1.0" encoding="utf-8"?>\n'
    xml += '<smil xmlns="http://www.w3.org/ns/SMIL" xmlns:epub="http://www.idpf.org/2007/ops" version="3.0">\n'
    xml += '  <body>\n'
    xml += f'    <seq id="{basename}_seq" epub:textref="{rel_text_path}">\n'

    audio_ref = f"../Audio/{audio_filename}"

    for i, seg in enumerate(segments):
        clip_start = seg['start']
        if i < len(segments) - 1:
            clip_end = segments[i+1]['start']
        else:
            clip_end = seg['end']

        t_start_str = format_time(clip_start)
        t_end_str = format_time(clip_end)
        t_id = seg['id']

        xml += f'      <par id="par_{t_id}">\n'
        xml += f'        <text src="{rel_text_path}#{t_id}"/>\n'
        xml += f'        <audio src="{audio_ref}" clipBegin="{t_start_str}" clipEnd="{t_end_str}"/>\n'
        xml += '      </par>\n'

    xml += '    </seq>\n'
    xml += '  </body>\n'
    xml += '</smil>'
    return xml


# -------------------------------------------------------------------------
# OPF & PACKAGING
# -------------------------------------------------------------------------
def update_opf(opf_path, manifest_additions, spine_mapping, total_durations):
    with open(opf_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'xml')

    for meta in soup.metadata.find_all("meta", property="media:duration"): meta.decompose()
    for meta in soup.metadata.find_all("meta", property="media:active-class"): meta.decompose()

    meta_dur = soup.new_tag("meta", property="media:duration")
    total_sec = sum(total_durations.values())
    m, s = divmod(total_sec, 60)
    h, m = divmod(m, 60)
    meta_dur.string = f"{int(h)}:{int(m):02d}:{s:06.3f}"
    soup.metadata.append(meta_dur)

    meta_active = soup.new_tag("meta", property="media:active-class")
    meta_active.string = "-epub-media-overlay-active"
    soup.metadata.append(meta_active)

    for smil_id, dur_sec in total_durations.items():
        meta = soup.new_tag("meta", property="media:duration")
        meta['refines'] = f"#{smil_id}"
        meta.string = format_time(dur_sec)
        soup.metadata.append(meta)

    manifest = soup.manifest
    for item in manifest_additions:
        existing = manifest.find("item", id=item['id'])
        if existing: existing.decompose()

    for item in manifest_additions:
        new_tag = soup.new_tag("item", **item)
        manifest.append(new_tag)

    for text_id, smil_id in spine_mapping.items():
        item = manifest.find("item", id=text_id)
        if item:
            item['media-overlay'] = smil_id

    xml_str = str(soup.prettify())
    xml_str = re.sub(r'(<meta property="media:active-class"[^>]*>)\s+([^\s<]+)\s+(</meta>)', r'\1\2\3', xml_str)
    xml_str = re.sub(r'(<opf:meta property="media:active-class"[^>]*>)\s+([^\s<]+)\s+(</opf:meta>)', r'\1\2\3', xml_str)

    with open(opf_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)

def zip_dir(folder, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(os.path.join(folder, "mimetype"), "mimetype", compress_type=zipfile.ZIP_STORED)
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file == "mimetype": continue
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, folder)
                zipf.write(abs_path, rel_path)
