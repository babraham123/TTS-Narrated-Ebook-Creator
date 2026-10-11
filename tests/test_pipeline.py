"""End-to-end runs of the CLI with the fake backend: resume, packaging, report, signals."""

import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import zipfile

import pytest
from bs4 import BeautifulSoup

from calliope.cache import ClipCache
from calliope.epub import format_clock
from conftest import EXPECTED_CHAPTERS, EXPECTED_SENTENCES, spawn_cli

CLOCK = re.compile(r"^(\d+):(\d\d):(\d\d)\.(\d\d\d)$")


def clock_ms(value):
    h, m, s, ms = map(int, CLOCK.match(value).groups())
    return ((h * 60 + m) * 60 + s) * 1000 + ms


def test_rerun_synthesizes_nothing_and_is_byte_identical(run_cli):
    first = run_cli("first.epub")
    second = run_cli("second.epub")
    fresh = run_cli("fresh.epub", work="other-work")

    assert (first.code, second.code, fresh.code) == (0, 0, 0)
    assert first.report["synthesized"] == first.report["unique_clips"]
    assert second.report["synthesized"] == 0
    assert second.report["attempts_this_run"] == 0
    assert second.report["cached"] == second.report["unique_clips"]
    assert first.bytes() == second.bytes() == fresh.bytes()


def test_rerun_does_not_load_the_model(run_cli, tmp_path, book, voice):
    from calliope.pipeline import Options, run

    run_cli("first.epub")

    def no_backend():
        raise AssertionError("model loaded although every clip is cached")

    opts = Options(input=book, output=str(tmp_path / "again.epub"), voice=voice, lang="en",
                   model="fake", work_dir=str(tmp_path / "work"))
    stats = run(opts, no_backend, log=lambda *_: None)
    assert stats["synthesized"] == 0
    assert (tmp_path / "again.epub").read_bytes() == (tmp_path / "first.epub").read_bytes()


def test_fake_backend_never_imports_torch(run_cli):
    assert run_cli().code == 0
    assert "torch" not in sys.modules
    assert "chatterbox" not in sys.modules


def test_package_structure(run_cli):
    result = run_cli()
    assert result.code == 0

    with zipfile.ZipFile(result.epub) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        assert names[0] == "mimetype"
        assert infos[0].compress_type == zipfile.ZIP_STORED
        assert z.read("mimetype") == b"application/epub+zip"
        assert names[1:] == sorted(names[1:])
        assert all(i.date_time == (1980, 1, 1, 0, 0, 0) for i in infos)
        assert not any("/." in n or n.startswith(".") for n in names)

    opf = BeautifulSoup(result.read("OEBPS/content.opf"), "xml")
    metas = opf.metadata.find_all("meta", property="media:duration")
    total = [m for m in metas if not m.get("refines")]
    per_overlay = {m["refines"].lstrip("#"): clock_ms(m.string) for m in metas if m.get("refines")}
    assert len(total) == 1
    assert clock_ms(total[0].string) == sum(per_overlay.values())
    assert opf.metadata.find("meta", property="media:active-class").string == "-epub-media-overlay-active"

    overlay_items = {i["id"]: i for i in opf.manifest.find_all("item", attrs={"media-overlay": True})}
    assert len(overlay_items) == EXPECTED_CHAPTERS
    assert set(per_overlay) == {i["media-overlay"] for i in overlay_items.values()}
    audio_items = opf.manifest.find_all("item", attrs={"media-type": "audio/mpeg"})
    assert len(audio_items) == EXPECTED_CHAPTERS  # one audio file per chapter

    css_item = opf.manifest.find("item", id="sync_css")
    assert ".-epub-media-overlay-active" in result.read("OEBPS/" + css_item["href"])

    spoken = 0
    for text_item in overlay_items.values():
        smil_item = opf.manifest.find("item", id=text_item["media-overlay"])
        assert smil_item["media-type"] == "application/smil+xml"
        smil = BeautifulSoup(result.read("OEBPS/" + smil_item["href"]), "xml")
        xhtml = result.read("OEBPS/" + text_item["href"])
        audio_srcs = set()
        previous_end = 0
        for par in smil.find_all("par"):
            text_src = par.find("text")["src"]
            audio = par.find("audio")
            fragment = text_src.split("#")[1]
            assert f'id="{fragment}"' in xhtml
            assert text_src.split("#")[0] == "../" + text_item["href"]
            begin, end = clock_ms(audio["clipBegin"]), clock_ms(audio["clipEnd"])
            assert begin == previous_end and end > begin  # contiguous, no holes or overlaps
            previous_end = end
            audio_srcs.add(audio["src"])
            spoken += 1
        assert previous_end == per_overlay[smil_item["id"]]
        assert len(audio_srcs) == 1
    assert spoken == EXPECTED_SENTENCES


def test_markup_is_preserved_and_escaped(run_cli):
    result = run_cli()
    xhtml = result.read("OEBPS/Text/ch1.xhtml")
    assert "Tom &amp; Jerry" in xhtml
    assert "&lt;suddenly&gt;" in xhtml
    assert 'epub:type="pagebreak"' in xhtml
    assert '<em class="loud">' in xhtml
    BeautifulSoup(xhtml, "xml")  # well-formed enough to parse


def test_audio_duration_matches_overlay(run_cli):
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe not installed")
    result = run_cli()
    opf = BeautifulSoup(result.read("OEBPS/content.opf"), "xml")
    expected = {m["refines"]: clock_ms(m.string)
                for m in opf.metadata.find_all("meta", property="media:duration") if m.get("refines")}
    with zipfile.ZipFile(result.epub) as z:
        z.extract("OEBPS/Audio/ch1.mp3", result.epub.parent / "x")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
         str(result.epub.parent / "x/OEBPS/Audio/ch1.mp3")],
        capture_output=True, text=True, check=True)
    assert abs(float(probe.stdout) * 1000 - expected["#smil_ch1"]) < 60


def test_report_contents(run_cli):
    result = run_cli("narrated.epub")
    assert result.report_path.name == "narrated.report.json"
    assert result.report_path.parent == result.epub.parent
    report = result.report
    assert report["status"] == "ok"
    assert report["model"] == "fake"
    assert report["chapters"] == EXPECTED_CHAPTERS
    assert report["sentences"] == EXPECTED_SENTENCES
    assert report["retries"] == 0
    assert report["flagged"] == [] and report["flagged_count"] == 0

    opf = BeautifulSoup(result.read("OEBPS/content.opf"), "xml")
    total = next(m for m in opf.metadata.find_all("meta", property="media:duration") if not m.get("refines"))
    assert report["audio_seconds"] == pytest.approx(clock_ms(total.string) / 1000, abs=0.001)
    assert report["real_time_factor"] == pytest.approx(
        report["wall_seconds"] / report["audio_seconds"], rel=0.01, abs=1e-4)
    assert "wall_seconds / audio_seconds" in report["real_time_factor_definition"]


def test_epubcheck_passes(run_cli):
    if shutil.which("epubcheck") is None:
        pytest.skip("epubcheck not installed")
    result = run_cli(env={"CALLIOPE_FORCE_SILENCE": "2:1"})
    check = subprocess.run(["epubcheck", str(result.epub)], capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
    assert "0 fatals / 0 errors" in check.stdout


def test_usage_errors_exit_2(run_cli):
    with pytest.raises(SystemExit) as turbo_fr:
        run_cli(extra=["--model", "turbo", "--lang", "fr"])
    assert turbo_fr.value.code == 2


def test_failure_exits_nonzero_with_report(run_cli, tmp_path):
    with open(run_cli.book, "wb") as f:
        f.write(b"not a zip")
    result = run_cli()
    assert result.code == 1
    assert result.report["status"] == "failed"
    assert "zip" in result.report["error"].lower()
    assert not result.epub.exists()


def _wait_for(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.mark.parametrize("kill_after_clips", [1, 4])
def test_sigterm_leaves_a_valid_cache_and_resume_matches_clean_run(run_cli, tmp_path, kill_after_clips):
    work = tmp_path / "work"
    args = run_cli.args("killed.epub", work="work")
    proc = spawn_cli(args, env={"CALLIOPE_FAKE_DELAY": "0.2"})

    def clips_done():
        return len(glob.glob(str(work / "clips/*/*.json"))) >= kill_after_clips

    assert _wait_for(clips_done, timeout=60), proc.communicate()
    proc.send_signal(signal.SIGTERM)
    started = time.monotonic()
    proc.communicate(timeout=10)
    assert time.monotonic() - started < 5  # exits promptly
    assert proc.returncode == 143

    assert not (tmp_path / "killed.epub").exists()
    killed_report = json.loads((tmp_path / "killed.report.json").read_text())
    assert killed_report["status"] == "terminated"
    leftovers = [p for p in glob.glob(str(work / "**/*"), recursive=True) if ".tmp-" in os.path.basename(p)]
    assert leftovers == []
    cache = ClipCache(str(work))
    keys = [os.path.basename(p)[:-5] for p in glob.glob(str(work / "clips/*/*.json"))]
    assert keys and all(cache.load(k) is not None for k in keys)

    resumed = run_cli("killed.epub")
    clean = run_cli("clean.epub", work="clean-work")
    assert resumed.code == 0
    assert resumed.report["cached"] >= kill_after_clips
    assert resumed.bytes() == clean.bytes()


def test_format_clock():
    assert format_clock(0) == "0:00:00.000"
    assert format_clock(3_723_004) == "1:02:03.004"
