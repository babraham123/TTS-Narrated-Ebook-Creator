"""Per-sentence quality checks for synthesized clips.

The cheap checks run on every attempt:
  * duration vs. what the text length predicts: much shorter suggests truncation, much
    longer suggests rambling or repetition;
  * silence: the share of 20 ms frames quieter than a threshold.
The optional Whisper check transcribes the chosen clip and computes the word error rate
(character error rate for zh/ja) against the sentence text. It only flags; it never retries.
"""

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np

from calliope.normalize import normalize_symbols

# Speaking rate in characters per second used to predict a clip's duration.
DEFAULT_CHARS_PER_SECOND = 14.0
CHARS_PER_SECOND_BY_LANG = {"zh": 4.5, "ja": 7.0, "ko": 7.0}
CHARACTER_LEVEL_LANGS = {"zh", "ja"}


@dataclass
class QAConfig:
    chars_per_second: float = DEFAULT_CHARS_PER_SECOND
    min_ratio: float = 0.4  # shorter than expected * min_ratio => truncated
    max_ratio: float = 2.5  # longer than expected * max_ratio + slack => rambling
    slack_seconds: float = 1.0  # absorbs fixed lead-in/out silence on very short sentences
    silence_db: float = -40.0  # frame RMS below this (dBFS) counts as silent
    max_silence: float = 0.6  # more silent frames than this share => mostly silence

    @classmethod
    def for_language(cls, lang, **overrides):
        cps = CHARS_PER_SECOND_BY_LANG.get(lang, DEFAULT_CHARS_PER_SECOND)
        values = {"chars_per_second": cps, **{k: v for k, v in overrides.items() if v is not None}}
        return cls(**values)


@dataclass
class QAResult:
    duration: float
    expected: float
    silence_fraction: float
    reasons: list = field(default_factory=list)
    badness: float = 0.0  # 0 when passed; larger is worse, used to pick the best attempt

    @property
    def passed(self):
        return not self.reasons


def silence_fraction(wav, sr, threshold_db, frame_seconds=0.02):
    frame = max(1, int(sr * frame_seconds))
    n = len(wav) // frame
    if n == 0:
        return 1.0
    frames = np.asarray(wav[: n * frame], dtype=np.float64).reshape(n, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    db = 20.0 * np.log10(rms + 1e-12)
    return float(np.mean(db < threshold_db))


def evaluate(wav, sr, text, cfg):
    duration = len(wav) / sr
    expected = len(text) / cfg.chars_per_second
    silent = silence_fraction(wav, sr, cfg.silence_db)
    result = QAResult(round(duration, 3), round(expected, 3), round(silent, 3))

    shortest = expected * cfg.min_ratio
    longest = expected * cfg.max_ratio + cfg.slack_seconds
    if duration < shortest:
        result.reasons.append("too_short")
        result.badness += (shortest - duration) / shortest
    if duration > longest:
        result.reasons.append("too_long")
        result.badness += (duration - longest) / longest
    if silent > cfg.max_silence:
        result.reasons.append("mostly_silence")
        result.badness += (silent - cfg.max_silence) / (1.0 - cfg.max_silence + 1e-9)
    return result


# -------------------------------------------------------------------------
# Whisper transcription check
# -------------------------------------------------------------------------
def _tokens(text, lang):
    # Whisper writes numbers as digits ("Chapter 1."); spell them out the way the
    # reference was, so "1" and "one" compare equal.
    text = normalize_symbols(unicodedata.normalize("NFKC", text), lang).lower()
    text = re.sub(r"[^\w\s']", " ", text).replace("'", "")
    if lang in CHARACTER_LEVEL_LANGS:
        return [c for c in text if not c.isspace()]
    return text.split()


def error_rate(reference, hypothesis, lang="en"):
    """Word error rate (character error rate for zh/ja): edit distance / reference length."""
    ref, hyp = _tokens(reference, lang), _tokens(hypothesis, lang)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


class WhisperTranscriber:
    """faster-whisper, imported lazily so it stays an optional dependency."""

    SAMPLE_RATE = 16000

    def __init__(self, model_size, lang, device="auto", compute_type="int8"):
        from faster_whisper import WhisperModel

        self.name = f"faster-whisper:{model_size}:{compute_type}"
        self.lang = lang
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, wav, sr):
        if sr != self.SAMPLE_RATE:
            n = int(round(len(wav) * self.SAMPLE_RATE / sr))
            wav = np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav)
        segments, _info = self.model.transcribe(
            np.asarray(wav, dtype=np.float32),
            language=self.lang,
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
