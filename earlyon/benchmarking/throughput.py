"""Throughput / latency benchmarking with proper CUDA sync.

Single-sample inference benchmark (batch=1) per v0.1 routing constraint. Use
``num_warmup>=50`` on CUDA; GPU clocks need ~30 iterations to stabilize.

``benchmark_wrapper`` runs a fixed random-noise input and is fast to
reproduce; trained heads may fire spuriously on noise, so it's a best-case
upper bound. ``benchmark_wrapper_on_loader`` drives real samples through the
wrapper and is the honest input-distribution signal — see the v0.2 README
appendix for the dual numbers.
"""

from __future__ import annotations

import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader

from earlyon.core.wrappers import EarlyExitWrapper

_Batch = tuple[torch.Tensor, torch.Tensor]


@dataclass
class BenchmarkResult:
    throughput_ips: float
    latency_median_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    avg_computation_used: float
    exit_distribution: dict[str, float]
    num_runs: int
    device: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _percentile(values: list[float], p: float) -> float:
    idx = max(0, min(len(values) - 1, int(round(p * (len(values) - 1)))))
    return sorted(values)[idx]


def benchmark_wrapper(
    model: EarlyExitWrapper,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = "cpu",
    num_warmup: int = 50,
    num_runs: int = 500,
) -> BenchmarkResult:
    """Benchmark a wrapper in inference mode."""
    if input_shape[0] != 1:
        raise ValueError("v0.1 inference benchmark requires batch_size=1")
    model = model.to(device).eval()
    dummy = torch.randn(input_shape, device=device)

    # warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy, mode="inference")
    _sync(device)

    latencies: list[float] = []
    exits: Counter[int] = Counter()
    comp_used: list[float] = []

    with torch.no_grad():
        wall_start = time.perf_counter()
        for _ in range(num_runs):
            t0 = time.perf_counter()
            result = model(dummy, mode="inference")
            _sync(device)
            latencies.append(time.perf_counter() - t0)
            exits[result.exit_taken] += 1
            comp_used.append(result.computation_used)
        wall_total = time.perf_counter() - wall_start

    # exit_distribution keys: "exit_0", "exit_1", ..., "final"
    def label(idx: int) -> str:
        return "final" if idx == -1 else f"exit_{idx}"

    dist = {label(k): v / num_runs for k, v in exits.items()}

    return BenchmarkResult(
        throughput_ips=num_runs / wall_total,
        latency_median_ms=statistics.median(latencies) * 1000,
        latency_p50_ms=_percentile(latencies, 0.50) * 1000,
        latency_p95_ms=_percentile(latencies, 0.95) * 1000,
        latency_p99_ms=_percentile(latencies, 0.99) * 1000,
        avg_computation_used=sum(comp_used) / len(comp_used),
        exit_distribution=dist,
        num_runs=num_runs,
        device=device,
    )


def benchmark_wrapper_on_loader(
    model: EarlyExitWrapper,
    loader: DataLoader[_Batch],
    device: str = "cpu",
    num_warmup: int = 50,
    num_runs: int = 500,
) -> BenchmarkResult:
    """Benchmark a wrapper in inference mode using real samples from a loader.

    The loader must yield (image, label) batches with ``batch_size == 1`` to
    match v0.2's single-sample routing. The first ``num_warmup`` samples are
    discarded; the next ``num_runs`` are timed. The loader is cycled if it
    runs out before ``num_warmup + num_runs`` samples have been drawn.
    """
    if loader.batch_size is None or loader.batch_size != 1:
        raise ValueError(
            f"benchmark_wrapper_on_loader requires loader.batch_size=1 "
            f"(got {loader.batch_size}); v0.2 inference is single-sample"
        )
    # an empty loader would make the cycling sample_stream spin forever; fail fast.
    try:
        is_empty = len(loader) == 0
    except TypeError:
        is_empty = False  # unsized (IterableDataset) — cannot pre-check
    if is_empty:
        raise ValueError("benchmark_wrapper_on_loader requires a non-empty loader")
    model = model.to(device).eval()

    def sample_stream() -> Iterator[torch.Tensor]:
        while True:
            for images, _targets in loader:
                yield images.to(device)

    stream = sample_stream()

    with torch.no_grad():
        for _ in range(num_warmup):
            x = next(stream)
            _ = model(x, mode="inference")
    _sync(device)

    latencies: list[float] = []
    exits: Counter[int] = Counter()
    comp_used: list[float] = []

    with torch.no_grad():
        wall_start = time.perf_counter()
        for _ in range(num_runs):
            x = next(stream)
            t0 = time.perf_counter()
            result = model(x, mode="inference")
            _sync(device)
            latencies.append(time.perf_counter() - t0)
            exits[result.exit_taken] += 1
            comp_used.append(result.computation_used)
        wall_total = time.perf_counter() - wall_start

    def label(idx: int) -> str:
        return "final" if idx == -1 else f"exit_{idx}"

    dist = {label(k): v / num_runs for k, v in exits.items()}

    return BenchmarkResult(
        throughput_ips=num_runs / wall_total,
        latency_median_ms=statistics.median(latencies) * 1000,
        latency_p50_ms=_percentile(latencies, 0.50) * 1000,
        latency_p95_ms=_percentile(latencies, 0.95) * 1000,
        latency_p99_ms=_percentile(latencies, 0.99) * 1000,
        avg_computation_used=sum(comp_used) / len(comp_used),
        exit_distribution=dist,
        num_runs=num_runs,
        device=device,
    )


def benchmark_backbone(
    backbone: torch.nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = "cpu",
    num_warmup: int = 50,
    num_runs: int = 500,
) -> BenchmarkResult:
    """Benchmark the raw backbone (no early-exit logic). Used as baseline."""
    backbone = backbone.to(device).eval()
    dummy = torch.randn(input_shape, device=device)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = backbone(dummy)
    _sync(device)

    latencies: list[float] = []
    with torch.no_grad():
        wall_start = time.perf_counter()
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = backbone(dummy)
            _sync(device)
            latencies.append(time.perf_counter() - t0)
        wall_total = time.perf_counter() - wall_start

    return BenchmarkResult(
        throughput_ips=num_runs / wall_total,
        latency_median_ms=statistics.median(latencies) * 1000,
        latency_p50_ms=_percentile(latencies, 0.50) * 1000,
        latency_p95_ms=_percentile(latencies, 0.95) * 1000,
        latency_p99_ms=_percentile(latencies, 0.99) * 1000,
        avg_computation_used=1.0,
        exit_distribution={"final": 1.0},
        num_runs=num_runs,
        device=device,
    )
