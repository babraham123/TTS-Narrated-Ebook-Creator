"""Clip quality checks: the retry path (driven by the forced-silence hook), the evaluation
rules, and the optional Whisper check (with a fake transcriber)."""

import glob
import json
import os

import numpy as np
import pytest
import soundfile as sf

from calliope.pipeline import Options, run
from calliope.qa import QAConfig, error_rate, evaluate
from calliope.tts import FakeBackend
from conftest import EXPECTED_SENTENCES

SR = 24000


def tone(seconds, amplitude=0.3):
    t = np.arange(int(SR * seconds)) / SR
    return (amplitude * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def test_retry_recovers_from_a_silent_first_attempt(run_cli):
    result = run_cli(env={"CALLIOPE_FORCE_SILENCE": "2:1"})
    report = result.report
    assert result.code == 0
    assert report["retries"] == 1
    assert report["attempts_this_run"] == EXPECTED_SENTENCES + 1
    assert report["flagged"] == []


def test_exhausted_retries_keep_best_attempt_and_flag_it(run_cli):
    result = run_cli(extra=["--max-retries", "3"], env={"CALLIOPE_FORCE_SILENCE": "2"})
    report = result.report
    assert result.code == 0
    assert report["retries"] == 3
    assert report["flagged_count"] == 1
    flagged = report["flagged"][0]
    assert flagged["index"] == 2
    assert flagged["attempts"] == 4
    assert flagged["reasons"] == ["mostly_silence"]
    assert flagged["silence_fraction"] == 1.0
    assert flagged["text"] == "The clocks were striking thirteen."

    # Same settings: the flagged clip is final, nothing is resynthesized.
    again = run_cli("again.epub", extra=["--max-retries", "3"], env={"CALLIOPE_FORCE_SILENCE": "2"})
    assert again.report["synthesized"] == 0
    assert again.report["flagged"] == report["flagged"]
    assert again.bytes() == result.bytes()

    # A larger retry budget continues after the attempts already made.
    more = run_cli("more.epub", extra=["--max-retries", "5"], env={"CALLIOPE_FORCE_SILENCE": "2"})
    assert more.report["synthesized"] == 1
    assert more.report["attempts_this_run"] == 2
    assert more.report["flagged"][0]["attempts"] == 6

    # Without the hook, the next attempt passes and the flag clears.
    fixed = run_cli("fixed.epub", extra=["--max-retries", "6"])
    assert fixed.report["attempts_this_run"] == 1
    assert fixed.report["flagged"] == []


@pytest.mark.parametrize(
    "wav, text, reasons",
    [
        (tone(2.0), "x" * 28, []),  # 2 s expected at 14 chars/s
        (tone(0.3), "x" * 70, ["too_short"]),  # 5 s expected
        (tone(20.0), "x" * 28, ["too_long"]),  # > 2 s * 2.5 + 1 s
        (np.zeros(SR * 2, dtype=np.float32), "x" * 28, ["mostly_silence"]),
        (
            np.concatenate([tone(0.5), np.zeros(SR * 2, dtype=np.float32)]),
            "x" * 28,
            ["mostly_silence"],
        ),
        (np.zeros(0, dtype=np.float32), "x" * 28, ["too_short", "mostly_silence"]),
    ],
)
def test_evaluate(wav, text, reasons):
    result = evaluate(wav, SR, text, QAConfig())
    assert result.reasons == reasons
    assert result.passed == (not reasons)
    assert (result.badness == 0) == (not reasons)


def test_worse_clips_score_worse():
    cfg = QAConfig()
    slightly = evaluate(tone(0.7), SR, "x" * 28, cfg)  # limit is 0.8 s
    badly = evaluate(tone(0.1), SR, "x" * 28, cfg)
    assert 0 < slightly.badness < badly.badness


def test_error_rate():
    assert (
        error_rate("The clocks were striking thirteen.", "the clocks were striking thirteen") == 0
    )
    assert error_rate("one two three four", "one two four") == pytest.approx(0.25)
    assert error_rate("你好世界", "你好", lang="zh") == pytest.approx(0.5)
    assert error_rate("Chapter One", "Chapter 1.") == 0  # Whisper's digits match number words


class FakeTranscriber:
    """Transcribes perfectly (by looking the clip up in the work dir), except for one sentence
    it mishears."""

    name = "fake-whisper"

    def __init__(self, work_dir, garble):
        self.work_dir = work_dir
        self.garble = garble
        self.loads = 0
        self.calls = 0

    def __call__(self):
        self.loads += 1
        return self

    def transcribe(self, wav, sr):
        self.calls += 1
        for meta_path in glob.glob(os.path.join(self.work_dir, "clips", "*", "*.json")):
            if meta_path.endswith(".transcript.json"):
                continue
            clip, _ = sf.read(meta_path[:-5] + ".flac", dtype="float32")
            if np.array_equal(clip, wav):
                with open(meta_path) as f:
                    text = json.load(f)["text"]
                return "completely different words" if text == self.garble else text
        raise AssertionError("clip not found in cache")


def test_whisper_check_flags_high_error_rate_and_caches_transcripts(tmp_path, book, voice):
    work = str(tmp_path / "work")
    opts = Options(
        input=book,
        output=str(tmp_path / "out.epub"),
        voice=voice,
        lang="en",
        model="fake",
        work_dir=work,
    )
    transcriber = FakeTranscriber(work, garble="A short one.")

    stats = run(opts, FakeBackend, transcriber, log=lambda *_: None)
    assert [(f["text"], f["reasons"]) for f in stats["flagged"]] == [("A short one.", ["high_wer"])]
    assert stats["flagged"][0]["wer"] == 1.0
    assert stats["flagged"][0]["transcript"] == "completely different words"
    assert transcriber.loads == 1 and transcriber.calls == EXPECTED_SENTENCES

    rerun = FakeTranscriber(work, garble="A short one.")
    stats_again = run(opts, FakeBackend, rerun, log=lambda *_: None)
    assert rerun.loads == 0 and rerun.calls == 0
    assert stats_again["flagged"] == stats["flagged"]
