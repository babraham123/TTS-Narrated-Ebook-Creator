"""TTS backends (Chatterbox Turbo/Multilingual and a fake tone generator) plus the
text chunking and crossfade helpers around them.

torch and chatterbox are imported only when a ChatterboxBackend is constructed, so the
rest of the pipeline (and the fake backend) runs without them installed.
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
        r"([.?!])\s",  # 1. End of sentence
        r"([:;])\s",  # 2. Semi-colons
        r"([,])\s",  # 3. Commas
        r"(\s)",  # 4. Any space (fallback)
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


# -------------------------------------------------------------------------
# BACKENDS
# -------------------------------------------------------------------------
MODELS = ("turbo", "multilingual")


def resolve_model(model, lang):
    """Default to Turbo for English and Multilingual otherwise; Turbo is English-only."""
    if model is None:
        return "turbo" if lang == "en" else "multilingual"
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {', '.join(MODELS)}")
    if model == "turbo" and lang != "en":
        raise ValueError(
            f"Chatterbox Turbo is English-only; use --model multilingual for --lang {lang}"
        )
    return model


class ChatterboxBackend:
    """Chatterbox Turbo (chatterbox.tts_turbo.ChatterboxTurboTTS) or Multilingual
    (chatterbox.mtl_tts.ChatterboxMultilingualTTS), conditioned once on the voice sample."""

    def __init__(self, model, lang, voice, device="auto"):
        import torch

        if model == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS as model_cls
        else:
            from chatterbox.mtl_tts import SUPPORTED_LANGUAGES
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS as model_cls

            if lang not in SUPPORTED_LANGUAGES:
                raise ValueError(
                    f"Chatterbox Multilingual does not support {lang!r}; "
                    f"supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}"
                )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._torch = torch
        self.name = model
        self.lang = lang
        self.device = device
        self.model = model_cls.from_pretrained(device=device)
        # Conditioning once instead of per generate() call avoids re-embedding the voice per chunk.
        self.model.prepare_conditionals(voice)
        self.sr = self.model.sr

    def generate(self, text, seed):
        torch = self._torch
        torch.manual_seed(seed)  # seeds CUDA as well; sampling in Chatterbox is torch-only
        with torch.inference_mode():
            if self.name == "turbo":
                wav = self.model.generate(text)
            else:
                wav = self.model.generate(text, language_id=self.lang)
        return wav.squeeze(0).float().cpu().numpy()


class FakeBackend:
    """Deterministic stand-in for tests and layout debugging: a tone whose length scales with
    the text. Needs neither torch nor a model download."""

    name = "fake"
    sr = SAMPLE_RATE
    chars_per_second = 14.0

    def __init__(self, delay=0.0):
        self.delay = delay

    def generate(self, text, seed):
        if self.delay:
            import time

            time.sleep(self.delay)
        rng = np.random.default_rng(seed)
        n = int(self.sr * (0.2 + len(text) / self.chars_per_second))
        freq = 160.0 + float(rng.integers(0, 160))
        t = np.arange(n, dtype=np.float64) / self.sr
        return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def synthesize_sentence(backend, text, seed):
    """Synthesize one sentence, splitting long text into chunks joined with a crossfade."""
    chunks = apply_ellipsis_to_chunks(smart_split_text(text))
    wavs = [backend.generate(chunk, (seed + i * 7919) % 2**32) for i, chunk in enumerate(chunks)]
    if len(wavs) == 1:
        return np.asarray(wavs[0], dtype=np.float32)
    return crossfade_audio_segments(wavs, sr=backend.sr)
