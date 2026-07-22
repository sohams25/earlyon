"""Throughput / latency benchmarking with identical boundaries per model.

Fairness contract
-----------------
:func:`benchmark_models` is the one measurement core. Every model in a
comparison sees **the same sample sequence** (preloaded once, before any
timing), the same warmup count, the same ``eval()`` + ``torch.inference_mode``
setup, the same device placement, and the same per-iteration synchronization.
Two boundaries are supported and always labelled on the result:

* ``"model-only"`` (default) — samples are moved to the device *before* the
  timed region; the measurement covers the forward pass (and, for a wrapper,
  its routing logic + host synchronization) only.
* ``"end-to-end"`` — samples stay on CPU and the timed region includes the
  host-to-device copy. Loader/preprocessing time is never measured by either
  boundary.

The legacy helpers (:func:`benchmark_wrapper`, :func:`benchmark_backbone`,
:func:`benchmark_wrapper_on_loader`) are thin wrappers over the core and keep
their signatures. Note that a wrapper benchmarked on random noise is a
best-case bound — trained heads may fire spuriously on noise — so prefer the
loader variants for honest input-distribution numbers.

Batch size 1 is the primary supported mode: it is the latency-sensitive edge
case earlyon targets, and the single-sample router requires it.
"""

from __future__ import annotations

import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from earlyon.core.types import Batch, exit_label
from earlyon.core.wrappers import EarlyExitWrapper

BOUNDARY_MODEL_ONLY = "model-only"
BOUNDARY_END_TO_END = "end-to-end"


@dataclass
class BenchmarkResult:
    """One model's measurement under the fairness contract.

    ``avg_estimated_flops_fraction`` is the average of the wrapper's static
    per-exit FLOPs estimate (1.0 for plain modules) — an estimate, not a
    measurement; see ``InferenceResult.estimated_backbone_flops_fraction``.
    ``accuracy`` is populated only when the samples came from a labelled
    loader.
    """

    throughput_ips: float
    latency_median_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    avg_estimated_flops_fraction: float
    exit_distribution: dict[str, float]
    num_runs: int
    device: str
    num_warmup: int = 0
    batch_size: int = 1
    input_shape: tuple[int, ...] = ()
    dtype: str = "torch.float32"
    boundary: str = BOUNDARY_MODEL_ONLY
    sync: str = "none"  # "cuda-synchronize-per-iteration" on CUDA devices
    input_source: str = "random-noise"  # or "loader"
    accuracy: float | None = None
    num_labelled_samples: int = 0

    @property
    def avg_computation_used(self) -> float:
        """Deprecated alias for ``avg_estimated_flops_fraction``."""
        return self.avg_estimated_flops_fraction

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["input_shape"] = list(self.input_shape)
        return d


@dataclass
class ComparisonResult:
    """Results of benchmarking several models on identical samples.

    ``speedup_vs`` divides two models' throughputs; it is only meaningful
    because the fairness contract guarantees identical inputs and boundaries.
    """

    results: dict[str, BenchmarkResult] = field(default_factory=dict)

    def speedup_vs(self, model: str, baseline: str) -> float:
        return self.results[model].throughput_ips / max(
            self.results[baseline].throughput_ips, 1e-12
        )


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _percentile(values: list[float], p: float) -> float:
    idx = max(0, min(len(values) - 1, int(round(p * (len(values) - 1)))))
    return sorted(values)[idx]


def _draw_samples(
    loader: DataLoader[Batch] | None,
    input_shape: tuple[int, int, int, int] | None,
    count: int,
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
    """Materialize the exact sample sequence every model will see.

    From a loader: cycle it until ``count`` samples are drawn (keeping
    labels). Without one: fixed-seed random noise, generated once.
    """
    if loader is not None:
        if loader.batch_size is None or loader.batch_size != 1:
            raise ValueError(
                f"benchmarking requires loader.batch_size=1 (got {loader.batch_size}); "
                "batch-1 latency is the primary supported mode"
            )
        try:
            if len(loader) == 0:
                raise ValueError("benchmarking requires a non-empty loader")
        except TypeError:
            pass  # unsized (IterableDataset) — cannot pre-check
        images: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        while len(images) < count:
            drew_any = False
            for x, y in loader:
                drew_any = True
                images.append(x)
                labels.append(y)
                if len(images) >= count:
                    break
            if not drew_any:
                raise ValueError("benchmarking requires a non-empty loader")
        return images, labels
    if input_shape is None:
        raise ValueError("provide either a loader or an input_shape")
    if input_shape[0] != 1:
        raise ValueError(
            f"benchmarking requires batch_size=1 (got input_shape={input_shape}); "
            "batch-1 latency is the primary supported mode"
        )
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(input_shape, generator=gen) for _ in range(count)], None


def _run_one(
    model: nn.Module,
    samples: list[torch.Tensor],
    labels: list[torch.Tensor] | None,
    device: str,
    num_warmup: int,
    num_runs: int,
    boundary: str,
    input_source: str,
) -> BenchmarkResult:
    is_wrapper = isinstance(model, EarlyExitWrapper)
    model = model.to(device).eval()

    if boundary == BOUNDARY_MODEL_ONLY:
        staged = [s.to(device) for s in samples]
    elif boundary == BOUNDARY_END_TO_END:
        staged = [s.cpu() for s in samples]  # timed region includes .to(device)
    else:
        raise ValueError(f"unknown boundary {boundary!r}")

    def forward(x: torch.Tensor) -> Any:
        if boundary == BOUNDARY_END_TO_END:
            x = x.to(device)
        if is_wrapper:
            return model(x, mode="inference")
        return model(x)

    with torch.inference_mode():
        for i in range(num_warmup):
            forward(staged[i])
    _sync(device)

    latencies: list[float] = []
    exits: Counter[int] = Counter()
    flops_fraction: list[float] = []
    correct = 0
    labelled = 0

    with torch.inference_mode():
        wall_start = time.perf_counter()
        for i in range(num_runs):
            x = staged[num_warmup + i]
            t0 = time.perf_counter()
            out = forward(x)
            _sync(device)
            latencies.append(time.perf_counter() - t0)
            if is_wrapper:
                exits[out.exit_taken] += 1
                flops_fraction.append(out.estimated_backbone_flops_fraction)
                prediction = out.prediction
            else:
                exits[-1] += 1
                flops_fraction.append(1.0)
                prediction = out
            if labels is not None:
                target = int(labels[num_warmup + i].item())
                correct += int(int(prediction.argmax(dim=-1).item()) == target)
                labelled += 1
        wall_total = time.perf_counter() - wall_start

    dist = {exit_label(k): v / num_runs for k, v in exits.items()}
    return BenchmarkResult(
        throughput_ips=num_runs / wall_total,
        latency_median_ms=statistics.median(latencies) * 1000,
        latency_p50_ms=_percentile(latencies, 0.50) * 1000,
        latency_p95_ms=_percentile(latencies, 0.95) * 1000,
        latency_p99_ms=_percentile(latencies, 0.99) * 1000,
        avg_estimated_flops_fraction=sum(flops_fraction) / len(flops_fraction),
        exit_distribution=dist,
        num_runs=num_runs,
        device=device,
        num_warmup=num_warmup,
        batch_size=1,
        input_shape=tuple(samples[0].shape),
        dtype=str(samples[0].dtype),
        boundary=boundary,
        sync="cuda-synchronize-per-iteration" if device.startswith("cuda") else "none",
        input_source=input_source,
        accuracy=(correct / labelled) if labelled else None,
        num_labelled_samples=labelled,
    )


def benchmark_models(
    models: Mapping[str, nn.Module],
    loader: DataLoader[Batch] | None = None,
    input_shape: tuple[int, int, int, int] | None = None,
    device: str = "cpu",
    num_warmup: int = 50,
    num_runs: int = 500,
    boundary: str = BOUNDARY_MODEL_ONLY,
    seed: int = 0,
) -> ComparisonResult:
    """Benchmark several models under identical boundaries and inputs.

    ``models`` maps a label to a module: the raw backbone, an
    :class:`EarlyExitWrapper` (routed via its inference path), a smaller
    static baseline, a quantised variant — anything callable on the samples.
    Typical usage::

        cmp = benchmark_models(
            {"backbone": model.backbone, "early_exit": model, "small": resnet18},
            loader=test_loader, device="cuda",
        )
        cmp.speedup_vs("early_exit", "backbone")

    The exact same preloaded sample sequence is fed to every model, in the
    same order, with the same warmup, boundary and synchronization. Samples
    come from ``loader`` (cycled; labels kept, accuracy reported) or, when no
    loader is given, from fixed-seed random noise (best-case bound: trained
    exit heads may fire spuriously on noise).
    """
    if num_warmup < 0:
        raise ValueError(f"num_warmup must be >= 0 (got {num_warmup})")
    if num_runs <= 0:
        raise ValueError(f"num_runs must be > 0 (got {num_runs})")
    if not models:
        raise ValueError("models must be non-empty")
    samples, labels = _draw_samples(loader, input_shape, num_warmup + num_runs, seed)
    input_source = "loader" if loader is not None else "random-noise"
    comparison = ComparisonResult()
    for name, model in models.items():
        comparison.results[name] = _run_one(
            model, samples, labels, device, num_warmup, num_runs, boundary, input_source
        )
    return comparison


# ---------------- legacy single-model helpers ----------------


def benchmark_wrapper(
    model: EarlyExitWrapper,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = "cpu",
    num_warmup: int = 50,
    num_runs: int = 500,
) -> BenchmarkResult:
    """Benchmark a wrapper on fixed-seed random noise (best-case bound)."""
    return benchmark_models(
        {"model": model},
        input_shape=input_shape,
        device=device,
        num_warmup=num_warmup,
        num_runs=num_runs,
    ).results["model"]


def benchmark_wrapper_on_loader(
    model: EarlyExitWrapper,
    loader: DataLoader[Batch],
    device: str = "cpu",
    num_warmup: int = 50,
    num_runs: int = 500,
) -> BenchmarkResult:
    """Benchmark a wrapper on real samples from a batch-1 loader.

    This is the honest input-distribution signal; accuracy over the timed
    samples is reported alongside speed.
    """
    return benchmark_models(
        {"model": model},
        loader=loader,
        device=device,
        num_warmup=num_warmup,
        num_runs=num_runs,
    ).results["model"]


def benchmark_backbone(
    backbone: torch.nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = "cpu",
    num_warmup: int = 50,
    num_runs: int = 500,
) -> BenchmarkResult:
    """Benchmark the raw backbone (no early-exit logic). Used as baseline."""
    return benchmark_models(
        {"model": backbone},
        input_shape=input_shape,
        device=device,
        num_warmup=num_warmup,
        num_runs=num_runs,
    ).results["model"]
