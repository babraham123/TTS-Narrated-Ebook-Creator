"""The batch entrypoint: sidecar validation and moving failed books out of the queue, with a
fake `calliope` on PATH."""

import importlib.util
import os
import signal
import sys
from pathlib import Path

import pytest

BATCH = Path(__file__).resolve().parent.parent / "batch" / "calliope_batch.py"

# Writes the output EPUB and its report, then exits with $FAKE_CALLIOPE_RC.
FAKE_CALLIOPE = """\
import os, pathlib, sys
out = pathlib.Path(sys.argv[2])
out.write_text("epub")
out.with_suffix(".report.json").write_text("{}")
sys.exit(int(os.environ.get("FAKE_CALLIOPE_RC", "0")))
"""


@pytest.fixture
def batch(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("calliope_batch", BATCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    narrate = tmp_path / "narrate"
    dirs = {
        "inbox": narrate / "inbox",
        "work": narrate / "work",
        "failed": narrate / "failed",
        "readalouds": tmp_path / "readalouds",
        "voices": tmp_path / "voices",
    }
    for name, path in dirs.items():
        monkeypatch.setattr(mod, name, path)
    mod.inbox.mkdir(parents=True)
    mod.voices.mkdir()
    (mod.voices / "default.wav").write_bytes(b"")

    mod.alerts = []
    monkeypatch.setattr(mod, "notify", lambda title, msg, _prio: mod.alerts.append((title, msg)))
    monkeypatch.setattr(signal, "signal", lambda *_: None)  # keep pytest's own handlers

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "calliope"
    script.write_text(f"#!{sys.executable}\n{FAKE_CALLIOPE}")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("READALOUDS_OWNER", f"{os.getuid()}:{os.getgid()}")
    return mod


def add_book(batch, title, mtime, sidecar=None):
    book = batch.inbox / f"{title}.epub"
    book.write_bytes(b"epub")
    if sidecar is not None:
        book.with_suffix(".json").write_text(sidecar)
    os.utime(book, (mtime, mtime))
    return book


@pytest.mark.parametrize(
    "sidecar",
    ['{"voice": 5}', "[]", '"books"', '{"lang": null}', '{"collection": ["books"]}'],
)
def test_sidecar_of_the_wrong_shape_is_a_value_error(batch, sidecar):
    book = add_book(batch, "Book", 1000, sidecar)
    with pytest.raises(ValueError):
        batch.options(book)


@pytest.mark.parametrize(
    "sidecar, env",
    [
        ('{"voice": 5}', {}),
        ("[]", {}),
        ("not json", {}),
        (None, {"FAKE_CALLIOPE_RC": "1"}),
        (None, {"PATH": "/nonexistent"}),
    ],
    ids=["voice-not-string", "not-an-object", "not-json", "calliope-fails", "calliope-missing"],
)
def test_failed_book_moves_to_failed_with_an_alert_and_the_next_one_runs(
    batch, monkeypatch, sidecar, env
):
    add_book(batch, "Bad", 1000, sidecar)
    add_book(batch, "Good", 2000)

    with monkeypatch.context() as m:
        for k, v in env.items():
            m.setenv(k, v)
        assert batch.main() == 1
    assert sorted(p.name for p in batch.inbox.iterdir()) == ["Good.epub"]
    assert (batch.failed / "Bad" / "Bad.epub").exists()
    assert (batch.failed / "Bad" / "Bad.json").exists() == (sidecar is not None)
    if "FAKE_CALLIOPE_RC" in env:
        assert (batch.failed / "Bad" / "Bad.readaloud.report.json").exists()
    assert len(batch.alerts) == 1
    title, msg = batch.alerts[0]
    assert title == "Narration failed" and msg.startswith("Bad: ")

    assert batch.main() == 0
    assert list(batch.inbox.iterdir()) == []
    delivered = batch.readalouds / "books" / "Good"
    assert sorted(p.name for p in delivered.iterdir()) == [
        "Good.readaloud.epub",
        "Good.readaloud.report.json",
    ]
    assert batch.alerts[1][0] == "Narration done"
