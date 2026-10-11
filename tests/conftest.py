"""Shared fixtures: a small valid EPUB, a voice sample, and helpers to run the CLI with the fake
TTS backend (--skip-audio), so the suite needs neither a GPU nor torch."""

import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import soundfile as sf

from calliope.cli import main
from epub_factory import make_epub

# Sentences in epub_factory.CHAPTERS, headings included: 6 in chapter one, 4 in chapter two.
EXPECTED_SENTENCES = 10
EXPECTED_CHAPTERS = 2


@pytest.fixture
def book(tmp_path):
    return str(make_epub(tmp_path / "book.epub"))


@pytest.fixture
def voice(tmp_path):
    path = tmp_path / "voice.wav"
    sr = 24000
    t = np.arange(sr * 6) / sr
    sf.write(path, (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)
    return str(path)


class Runner:
    """Runs `calliope IN OUT --voice V --skip-audio --work-dir W` in-process."""

    def __init__(self, tmp_path, book, voice, monkeypatch):
        self.tmp_path = tmp_path
        self.book = book
        self.voice = voice
        self.monkeypatch = monkeypatch

    def args(self, out_name, work="work", extra=()):
        out = str(self.tmp_path / out_name)
        return [self.book, out, "--voice", self.voice, "--skip-audio",
                "--work-dir", str(self.tmp_path / work), *extra]

    def __call__(self, out_name="out.epub", work="work", extra=(), env=None):
        self.monkeypatch.delenv("CALLIOPE_FORCE_SILENCE", raising=False)
        for k, v in (env or {}).items():
            self.monkeypatch.setenv(k, v)
        code = main(self.args(out_name, work, extra))
        out = self.tmp_path / out_name
        return Result(code, out, out.with_suffix(".report.json"))


class Result:
    def __init__(self, code, epub, report_path):
        self.code = code
        self.epub = epub
        self.report_path = report_path

    @property
    def report(self):
        return json.loads(self.report_path.read_text())

    def bytes(self):
        return self.epub.read_bytes()

    def read(self, name):
        with zipfile.ZipFile(self.epub) as z:
            return z.read(name).decode("utf-8")


@pytest.fixture
def run_cli(tmp_path, book, voice, monkeypatch):
    return Runner(tmp_path, book, voice, monkeypatch)


def spawn_cli(args, env=None):
    """Start the CLI in a child process (isolated mode) for signal tests."""
    full_env = {**os.environ, **(env or {})}
    return subprocess.Popen([sys.executable, "-I", "-m", "calliope", *args], env=full_env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
