"""Paces GPU work by temperature: before each clip, wait while the GPU is too hot.

A passively cooled card (Tesla P4) in a case without airflow overheats and drops off the bus
under sustained load; its own throttling cannot hold it below that. Pausing between clips
costs only time, since every finished clip is already cached.
"""

import os
import subprocess
import time

PAUSE_ENV = "GPU_PAUSE_TEMP"
RESUME_ENV = "GPU_RESUME_TEMP"
DEFAULT_PAUSE = 80
DEFAULT_RESUME = 70
POLL_SECONDS = 10


def nvidia_smi_temperature():
    """The hottest GPU's temperature in °C."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    ).stdout
    return max(int(line) for line in out.split())


def _log(msg):
    # Flushed so a pause shows in the log as it starts, even with stdout block-buffered.
    print(msg, flush=True)


class GpuGuard:
    """wait() returns once the GPU is cool enough: at once at or below `pause` °C; above it,
    only after the temperature has dropped below `resume` °C. If the temperature cannot be
    read, it warns once and stops guarding."""

    def __init__(self, pause, resume, read=nvidia_smi_temperature, sleep=time.sleep, log=_log):
        if resume > pause:
            raise ValueError(f"{RESUME_ENV} ({resume}) must not exceed {PAUSE_ENV} ({pause})")
        self.pause, self.resume = pause, resume
        self.read, self.sleep, self.log = read, sleep, log
        self.enabled = True
        self.paused_seconds = 0.0

    @classmethod
    def from_env(cls, env=os.environ, **kwargs):
        try:
            pause = int(env.get(PAUSE_ENV, DEFAULT_PAUSE))
            resume = int(env.get(RESUME_ENV, DEFAULT_RESUME))
        except ValueError:
            raise ValueError(f"{PAUSE_ENV} and {RESUME_ENV} must be whole degrees Celsius")
        return cls(pause, resume, **kwargs)

    def _temperature(self):
        if not self.enabled:
            return None
        try:
            return self.read()
        except Exception as e:
            self.enabled = False
            self.log(f"warning: cannot read the GPU temperature ({e}); not pacing by temperature")
            return None

    def wait(self):
        temp = self._temperature()
        if temp is None or temp <= self.pause:
            return
        self.log(f"GPU at {temp} °C, above {self.pause} °C: pausing until below {self.resume} °C")
        started = time.monotonic()
        while temp is not None and temp >= self.resume:
            self.sleep(POLL_SECONDS)
            temp = self._temperature()
        paused = time.monotonic() - started
        self.paused_seconds += paused
        cooled = "temperature unknown" if temp is None else f"GPU at {temp} °C"
        self.log(f"{cooled}: resuming after {paused / 60:.1f} min")
