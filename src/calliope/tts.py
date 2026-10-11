"""Chatterbox TTS wrapper plus the text chunking and crossfade helpers around it.

torch and chatterbox are imported lazily so the EPUB tooling can be used without them.
"""

import re

import numpy as np

# === TUNING CONSTANTS ===
CHUNK_CHAR_LIMIT = 400
CROSSFADE_MS = 50
SAMPLE_RATE = 24000
# ========================


def smart_split_text(text, max_chars=CHUNK_CHAR_LIMIT):
    """
    Splits text while preserving prosody.
    Prioritizes splitting at:
    1. Strong punctuation (.?!)
    2. Semi-strong punctuation (:;)
    3. Weak punctuation (,)
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    # Define split priorities (Regex Pattern)
    split_patterns = [
        r'([.?!])\s',      # 1. End of sentence
        r'([:;])\s',       # 2. Semi-colons
        r'([,])\s',        # 3. Commas
        r'(\s)'            # 4. Any space (fallback)
    ]

    best_split_point = -1
    search_start = max_chars // 4
    search_end = max_chars

    chunk = text[:search_end]

    for pattern in split_patterns:
        matches = list(re.finditer(pattern, chunk))
        if matches:
            valid_matches = [m for m in matches if m.start() > search_start]
            if valid_matches:
                last_match = valid_matches[-1]
                best_split_point = last_match.end()
                break

    if best_split_point == -1:
        best_split_point = max_chars

    part1 = text[:best_split_point].strip()
    part2 = text[best_split_point:].strip()

    return [part1] + smart_split_text(part2, max_chars)


def apply_ellipsis_to_chunks(chunks):
    """Adds '...' to cut boundaries to improve TTS prosody."""
    modified_chunks = []
    for i, chunk in enumerate(chunks):
        new_chunk = chunk
        if i < len(chunks) - 1:
            if not new_chunk.strip()[-1] in ".?!":
                new_chunk = new_chunk + "..."
        if i > 0:
            new_chunk = "..." + new_chunk
        modified_chunks.append(new_chunk)
    return modified_chunks


def crossfade_audio_segments(wav_list, fade_ms=CROSSFADE_MS, sr=SAMPLE_RATE):
    """Stitches mono float audio arrays with a linear crossfade."""
    if not wav_list:
        return np.zeros(0, dtype=np.float32)
    fade = int(sr * fade_ms / 1000)
    combined = np.asarray(wav_list[0], dtype=np.float32)
    for next_wav in wav_list[1:]:
        next_wav = np.asarray(next_wav, dtype=np.float32)
        n = min(fade, len(combined), len(next_wav))
        if n == 0:
            combined = np.concatenate([combined, next_wav])
            continue
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        overlap = combined[-n:] * (1.0 - ramp) + next_wav[:n] * ramp
        combined = np.concatenate([combined[:-n], overlap, next_wav[n:]])
    return combined


def check_cuda():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_chatterbox(device):
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except ImportError:
        raise SystemExit("Error: Chatterbox not found. Please run: pip install chatterbox-tts")
    return ChatterboxMultilingualTTS.from_pretrained(device=device)


def generate_tts_chunk_chatterbox(model, text, speaker, lang):
    """
    Basic wrapper for Chatterbox generation of a single chunk.
    """
    clean_chunk = text.replace("\n", " ").strip()
    if not clean_chunk: return np.zeros(24000)

    try:
        wav = model.generate(
            text=clean_chunk,
            language_id=lang,
            audio_prompt_path=speaker
        )
        if hasattr(wav, "cpu"):
            wav = wav.cpu().numpy()
        return wav.flatten()

    except Exception as e:
        print(f"\n[Chatterbox Error] Failed on '{clean_chunk[:20]}...': {e}")
        return np.zeros(24000)
