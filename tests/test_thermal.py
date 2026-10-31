"""The GPU temperature guard: pausing between clips while the GPU is hot."""

import json
import signal
import time

import pytest

from calliope import thermal
from calliope.thermal import GpuGuard
from conftest import EXPECTED_SENTENCES, spawn_cli


class Temps:
    """A scripted temperature reader; the last reading repeats."""

    def __init__(self, *values):
        self.values = list(values)
        self.reads = 0

    def __call__(self):
        self.reads += 1
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]


def guard(read, pause=80, resume=70):
    logs, sleeps = [], []
    g = GpuGuard(pause, resume, read=read, sleep=sleeps.append, log=logs.append)
    return g, logs, sleeps


def test_hot_gpu_pauses_until_below_resume_and_logs_once_each_way():
    g, logs, sleeps = guard(Temps(93, 88, 75, 70, 69, 95))
    g.wait()
    assert len(sleeps) == 4  # polls at 88, 75, 70 and 69 °C
    assert len(logs) == 2
    assert "93 °C" in logs[0] and "pausing" in logs[0]
    assert "69 °C" in logs[1] and "resuming" in logs[1]


def test_at_or_below_pause_temperature_runs_on():
    g, logs, sleeps = guard(Temps(80))
    g.wait()
    g.wait()
    assert (logs, sleeps) == ([], [])


def test_unreadable_temperature_warns_once_and_stops_guarding():
    def broken():
        broken.calls += 1
        raise FileNotFoundError("nvidia-smi")

    broken.calls = 0
    g, logs, sleeps = guard(broken)
    for _ in range(3):
        g.wait()
    assert broken.calls == 1
    assert sleeps == []
    assert len(logs) == 1 and logs[0].startswith("warning:")


def test_reader_failing_during_a_pause_ends_the_pause():
    readings = iter([90])

    def flaky():
        return next(readings)  # StopIteration on the second read

    g, logs, sleeps = guard(flaky)
    g.wait()
    assert len(sleeps) == 1
    assert logs[1].startswith("warning:") and "resuming" in logs[2]
    assert not g.enabled


def test_env_defaults_and_validation():
    g = GpuGuard.from_env({})
    assert (g.pause, g.resume) == (80, 70)
    g = GpuGuard.from_env({"GPU_PAUSE_TEMP": "75", "GPU_RESUME_TEMP": "60"})
    assert (g.pause, g.resume) == (75, 60)
    with pytest.raises(ValueError):
        GpuGuard.from_env({"GPU_PAUSE_TEMP": "70", "GPU_RESUME_TEMP": "75"})
    with pytest.raises(ValueError):
        GpuGuard.from_env({"GPU_PAUSE_TEMP": "hot"})


def test_cli_pauses_between_clips_and_reports_the_pause(
    run_cli, fake_nvidia_smi, monkeypatch, capsys
):
    monkeypatch.setattr(thermal, "POLL_SECONDS", 0.01)
    fake_nvidia_smi(50, 50, 85, 78, 65, 50)
    result = run_cli()
    out = capsys.readouterr().out
    assert result.code == 0
    assert out.count("pausing") == 1 and "85 °C" in out
    assert out.count("resuming") == 1 and "65 °C" in out
    assert result.report["synthesized"] == result.report["unique_clips"]
    assert result.report["gpu_pause_seconds"] > 0


def test_cli_with_a_failing_nvidia_smi_warns_once(run_cli, fake_nvidia_smi, capsys):
    # No temperatures given: the fake nvidia-smi exits nonzero.
    result = run_cli()
    out = capsys.readouterr().out
    assert result.code == 0
    assert out.count("cannot read the GPU temperature") == 1
    assert result.report["gpu_pause_seconds"] == 0


def test_bad_env_is_a_usage_error(run_cli):
    with pytest.raises(SystemExit) as e:
        run_cli(env={"GPU_PAUSE_TEMP": "60", "GPU_RESUME_TEMP": "70"})
    assert e.value.code == 2


def test_sigterm_during_a_pause_exits_promptly_and_keeps_the_cache(
    run_cli, tmp_path, fake_nvidia_smi
):
    fake_nvidia_smi(50, 50, 95)
    proc = spawn_cli(run_cli.args("hot.epub"))
    seen = []
    for line in proc.stdout:
        seen.append(line)
        if "pausing" in line:
            break
    assert any("pausing" in line for line in seen), "".join(seen) + proc.stderr.read()
    time.sleep(0.2)
    proc.send_signal(signal.SIGTERM)
    started = time.monotonic()
    proc.communicate(timeout=10)
    assert time.monotonic() - started < 5
    assert proc.returncode == 143

    report = json.loads((tmp_path / "hot.report.json").read_text())
    assert report["status"] == "terminated"
    assert not (tmp_path / "hot.epub").exists()

    fake_nvidia_smi(50)
    resumed = run_cli("hot.epub")
    assert resumed.code == 0
    assert resumed.report["cached"] == 2
    assert resumed.report["sentences"] == EXPECTED_SENTENCES
