"""NVIDIA Jetson power/temperature profiling via tegrastats.

tegrastats blocks forever without an end condition, so we spawn it as a
background subprocess, read lines via a thread, and tear it down explicitly.
On non-Jetson systems all profile calls return empty telemetry (``None``
fields) so the rest of the benchmark suite still runs — missing data is
reported as missing, never invented as zeros.

Power/energy semantics (three distinct quantities, never conflated):

* **instantaneous power** — one tegrastats sample (``JetsonSample.power_mw``,
  ``JetsonRun.power_mw``): the module power at that sampling instant.
* **average power over a window** — the mean of the instantaneous samples
  that fall inside a benchmark window (``EnergySummary.avg_power_mw``).
* **integrated energy** — the trapezoidal integral of instantaneous power
  over the window (``EnergySummary.energy_mj``). Energy-per-inference is only
  reported when this integral exists and the inference count is known.
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
    """One parsed tegrastats line. ``None`` means the field was absent from
    the line (formats differ across JetPack releases) — not zero."""

    timestamp: float
    gpu_util_pct: Optional[float]
    temp_c: Optional[float]
    power_mw: Optional[float]


@dataclass
class JetsonRun:
    """Per-inference telemetry. Power/temp/util are the *latest instantaneous*
    tegrastats sample at the time the inference finished — not an average and
    not an energy; see :class:`EnergySummary` for integrated quantities."""

    latency_ms: float
    exit_taken: int
    confidence: float
    estimated_backbone_flops_fraction: float
    gpu_util_pct: Optional[float]
    temp_c: Optional[float]
    power_mw: Optional[float]

    @property
    def computation_used(self) -> float:
        """Deprecated alias for ``estimated_backbone_flops_fraction``."""
        return self.estimated_backbone_flops_fraction


@dataclass
class EnergySummary:
    """Power/energy over one benchmark window.

    ``energy_mj`` is the trapezoidal integral of the instantaneous power
    samples inside the window; it (and ``energy_per_inference_mj``) is
    ``None`` when fewer than two power samples landed in the window — a
    single sample cannot be integrated, and inventing a rectangle would be a
    fabricated measurement.
    """

    num_inferences: int
    window_seconds: float
    sampling_interval_ms: int
    num_power_samples: int
    avg_power_mw: Optional[float]
    energy_mj: Optional[float]
    energy_per_inference_mj: Optional[float]


_GR3D_RE = re.compile(r"GR3D_FREQ\s+(\d+)%")
_TEMP_RE = re.compile(r"(?:CPU|GPU|tj|pll|thermal)@(\d+(?:\.\d+)?)C", re.IGNORECASE)
_POWER_RE = re.compile(r"(?:POM_5V_IN|VDD_IN|VDD_GPU|VDD_CPU_GPU_CV)\s+(\d+)")


def _parse_line(line: str) -> JetsonSample:
    gpu = _GR3D_RE.search(line)
    temp = _TEMP_RE.search(line)
    power = _POWER_RE.search(line)
    return JetsonSample(
        timestamp=time.monotonic(),
        gpu_util_pct=float(gpu.group(1)) if gpu else None,
        temp_c=float(temp.group(1)) if temp else None,
        power_mw=float(power.group(1)) if power else None,
    )


def integrate_energy(
    samples: list[JetsonSample],
    window_start: float,
    window_end: float,
    num_inferences: int,
    sampling_interval_ms: int,
) -> EnergySummary:
    """Integrate instantaneous power samples over a benchmark window.

    Pure function (unit-testable without hardware). Only samples with a
    parsed power value that fall inside ``[window_start, window_end]`` are
    used; trapezoidal integration over their timestamps yields millijoules.
    """
    window_seconds = max(window_end - window_start, 0.0)
    in_window = [
        s for s in samples if s.power_mw is not None and window_start <= s.timestamp <= window_end
    ]
    avg_power: Optional[float] = None
    energy_mj: Optional[float] = None
    per_inference: Optional[float] = None
    if in_window:
        powers = [s.power_mw for s in in_window if s.power_mw is not None]
        avg_power = sum(powers) / len(powers)
    if len(in_window) >= 2:
        energy = 0.0  # mW * s == mJ
        for a, b in zip(in_window, in_window[1:]):
            assert a.power_mw is not None and b.power_mw is not None
            energy += 0.5 * (a.power_mw + b.power_mw) * (b.timestamp - a.timestamp)
        energy_mj = energy
        if num_inferences > 0:
            per_inference = energy / num_inferences
    return EnergySummary(
        num_inferences=num_inferences,
        window_seconds=window_seconds,
        sampling_interval_ms=sampling_interval_ms,
        num_power_samples=len(in_window),
        avg_power_mw=avg_power,
        energy_mj=energy_mj,
        energy_per_inference_mj=per_inference,
    )


class TegrastatsMonitor:
    """Spawn tegrastats, parse lines in a reader thread, terminate cleanly.

    Reusable: ``start()`` after ``stop()`` begins a fresh session (the stop
    event is cleared, the sample buffer is reset). A second ``start()`` while
    running is a no-op rather than a duplicate process/thread.
    """

    def __init__(self, interval_ms: int = 100, max_samples: int = 4096) -> None:
        if interval_ms <= 0:
            raise ValueError(f"interval_ms must be > 0 (got {interval_ms})")
        if max_samples <= 0:
            raise ValueError(f"max_samples must be > 0 (got {max_samples})")
        self.interval_ms = interval_ms
        self._max_samples = max_samples
        self._samples: deque[JetsonSample] = deque(maxlen=max_samples)
        self._proc: Optional[subprocess.Popen[str]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.available = shutil.which("tegrastats") is not None

    @property
    def running(self) -> bool:
        return self._proc is not None

    def start(self) -> None:
        if not self.available or self._proc is not None:
            return
        # a fresh session: clear the previous session's stop flag and samples.
        # Without this, a restarted monitor's reader thread exited immediately
        # (the stop event stayed set) and silently collected nothing.
        self._stop.clear()
        self._samples = deque(maxlen=self._max_samples)
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError:
            # binary disappeared or failed to spawn: degrade to unavailable
            self._proc = None
            self.available = False
            return
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            if self._stop.is_set():
                break
            self._samples.append(_parse_line(line))

    def latest(self) -> JetsonSample:
        if not self._samples:
            return JetsonSample(time.monotonic(), None, None, None)
        return self._samples[-1]

    def samples(self) -> list[JetsonSample]:
        """Snapshot of the current session's samples."""
        return list(self._samples)

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2)
            if self._proc.stdout is not None:
                self._proc.stdout.close()
        # join the reader so we don't drop a live thread holding the pipe
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._proc = None
        self._thread = None


class JetsonProfiler:
    """Profile a wrapper on Jetson and produce per-run telemetry.

    If tegrastats is not available (non-Jetson host), profiling still runs
    but power/temp/util fields are ``None``. This keeps tests and CI usable
    on a dev laptop without fabricating hardware numbers.
    """

    def __init__(self, interval_ms: int = 100) -> None:
        self.monitor = TegrastatsMonitor(interval_ms=interval_ms)

    def profile(
        self,
        model: EarlyExitWrapper,
        input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
        num_warmup: int = 50,
        num_runs: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> list[JetsonRun]:
        runs, _ = self.profile_with_energy(
            model, input_shape=input_shape, num_warmup=num_warmup, num_runs=num_runs, device=device
        )
        return runs

    def profile_with_energy(
        self,
        model: EarlyExitWrapper,
        input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
        num_warmup: int = 50,
        num_runs: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> tuple[list[JetsonRun], EnergySummary]:
        """Per-run telemetry plus power integrated over the timed window.

        The energy summary covers only the timed loop (warmup excluded) and
        divides by the true inference count; on hosts without tegrastats it
        reports zero samples and ``None`` energy.
        """
        if num_warmup < 0:
            raise ValueError(f"num_warmup must be >= 0 (got {num_warmup})")
        if num_runs <= 0:
            raise ValueError(f"num_runs must be > 0 (got {num_runs})")
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
            window_start = time.monotonic()
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
                            estimated_backbone_flops_fraction=(
                                result.estimated_backbone_flops_fraction
                            ),
                            gpu_util_pct=sample.gpu_util_pct,
                            temp_c=sample.temp_c,
                            power_mw=sample.power_mw,
                        )
                    )
            window_end = time.monotonic()
            summary = integrate_energy(
                self.monitor.samples(),
                window_start,
                window_end,
                num_inferences=num_runs,
                sampling_interval_ms=self.monitor.interval_ms,
            )
            return runs, summary
        finally:
            self.monitor.stop()
