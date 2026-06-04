"""NVIDIA Jetson power/temperature profiling via tegrastats.

tegrastats blocks forever without an end condition, so we spawn it as a
background subprocess, read lines via threads, and tear it down explicitly.
On non-Jetson systems all profile calls return zeros so the rest of the
benchmark suite still runs.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch

from earlyon.core.wrappers import EarlyExitWrapper


@dataclass
class JetsonSample:
    timestamp: float
    gpu_util_pct: float
    temp_c: float
    power_mw: float


@dataclass
class JetsonRun:
    latency_ms: float
    exit_taken: int
    confidence: float
    computation_used: float
    gpu_util_pct: float
    temp_c: float
    power_mw: float


_GR3D_RE = re.compile(r"GR3D_FREQ\s+(\d+)%")
_TEMP_RE = re.compile(r"(?:CPU|GPU|tj|pll|thermal)@(\d+(?:\.\d+)?)C", re.IGNORECASE)
_POWER_RE = re.compile(r"(?:POM_5V_IN|VDD_IN|VDD_GPU|VDD_CPU_GPU_CV)\s+(\d+)")


def _parse_line(line: str) -> JetsonSample:
    gpu = _GR3D_RE.search(line)
    temp = _TEMP_RE.search(line)
    power = _POWER_RE.search(line)
    return JetsonSample(
        timestamp=time.monotonic(),
        gpu_util_pct=float(gpu.group(1)) if gpu else 0.0,
        temp_c=float(temp.group(1)) if temp else 0.0,
        power_mw=float(power.group(1)) if power else 0.0,
    )


class TegrastatsMonitor:
    """Spawn tegrastats, parse the latest line on demand, terminate cleanly."""

    def __init__(self, interval_ms: int = 100, max_samples: int = 256) -> None:
        self.interval_ms = interval_ms
        self._samples: deque[JetsonSample] = deque(maxlen=max_samples)
        self._proc: Optional[subprocess.Popen[str]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.available = shutil.which("tegrastats") is not None

    def start(self) -> None:
        if not self.available:
            return
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            self._samples.append(_parse_line(line))

    def latest(self) -> JetsonSample:
        if not self._samples:
            return JetsonSample(time.monotonic(), 0.0, 0.0, 0.0)
        return self._samples[-1]

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        # join the reader so we don't drop a live thread holding the pipe
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._proc = None
        self._thread = None


class JetsonProfiler:
    """Profile a wrapper on Jetson and produce per-run telemetry.

    If tegrastats is not available (non-Jetson host), profiling still runs
    but power/temp/util fields are zero. This keeps tests and CI usable on
    a dev laptop.
    """

    def __init__(self) -> None:
        self.monitor = TegrastatsMonitor()

    def profile(
        self,
        model: EarlyExitWrapper,
        input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
        num_warmup: int = 50,
        num_runs: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> list[JetsonRun]:
        model = model.to(device).eval()
        dummy = torch.randn(input_shape, device=device)

        self.monitor.start()
        try:
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = model(dummy, mode="inference")
            if device.startswith("cuda"):
                torch.cuda.synchronize()

            runs: list[JetsonRun] = []
            with torch.no_grad():
                for _ in range(num_runs):
                    t0 = time.perf_counter()
                    result = model(dummy, mode="inference")
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    latency = time.perf_counter() - t0
                    sample = self.monitor.latest()
                    runs.append(
                        JetsonRun(
                            latency_ms=latency * 1000,
                            exit_taken=result.exit_taken,
                            confidence=result.confidence,
                            computation_used=result.computation_used,
                            gpu_util_pct=sample.gpu_util_pct,
                            temp_c=sample.temp_c,
                            power_mw=sample.power_mw,
                        )
                    )
            return runs
        finally:
            self.monitor.stop()
