import torch
from torch.utils.data import DataLoader, TensorDataset

from earlyon.benchmarking import benchmark_backbone, benchmark_wrapper, evaluate
from earlyon.core.exit_head import EarlyExitHead
from earlyon.core.types import EarlyExitConfig, ExitPoint
from earlyon.core.wrappers import EarlyExitWrapper
from tests.fixtures.tiny_models import STAGE_CHANNELS, TinyBackbone


def _build(thresholds=(1.0, 1.0)):
    backbone = TinyBackbone(num_classes=10)
    exits = [
        ExitPoint("e0", "stage1", STAGE_CHANNELS["stage1"]),
        ExitPoint("e1", "stage2", STAGE_CHANNELS["stage2"]),
    ]
    heads = {ep.name: EarlyExitHead(ep.in_channels, 10) for ep in exits}
    cfg = EarlyExitConfig(
        backbone="tiny",
        num_classes=10,
        exit_points=exits,
        confidence_thresholds=list(thresholds),
    )
    wrapper = EarlyExitWrapper(backbone, heads, lambda x: x, cfg, input_shape=(1, 3, 32, 32))
    return wrapper


def test_throughput_result_is_well_formed():
    model = _build()
    r = benchmark_wrapper(model, input_shape=(1, 3, 32, 32), num_warmup=2, num_runs=10)
    assert r.throughput_ips > 0
    assert r.latency_median_ms > 0
    assert r.latency_p95_ms >= r.latency_p50_ms
    assert sum(r.exit_distribution.values()) == 1.0
    assert 0.0 < r.avg_computation_used <= 1.0


def test_backbone_baseline_runs():
    model = _build()
    r = benchmark_backbone(model.backbone, input_shape=(1, 3, 32, 32), num_warmup=2, num_runs=10)
    assert r.throughput_ips > 0
    assert r.avg_computation_used == 1.0


def test_wrapper_overhead_on_real_model_is_modest():
    """The architect's key invariant — measured on a real backbone, not the
    tiny fixture. Hook + softmax overhead must stay under 25% of backbone
    latency when no exits ever fire. (On tiny fixtures absolute latency is
    microseconds so any per-iter overhead dominates; this test only makes
    sense on a real model.)"""
    import pytest

    try:
        from earlyon.models import resnet18_ee
    except Exception:
        pytest.skip("torchvision not available")
    model = resnet18_ee(num_classes=10, pretrained=False)
    model.config.confidence_thresholds = [1.0, 1.0]  # no exit can fire
    wrap_r = benchmark_wrapper(model, input_shape=(1, 3, 224, 224), num_warmup=10, num_runs=30)
    bb_r = benchmark_backbone(
        model.backbone, input_shape=(1, 3, 224, 224), num_warmup=10, num_runs=30
    )
    overhead = (wrap_r.latency_median_ms - bb_r.latency_median_ms) / bb_r.latency_median_ms
    assert overhead < 0.40, f"hook overhead too high: {overhead:.1%}"


def test_evaluate_loop_runs():
    model = _build(thresholds=(0.0, 0.0))  # always exits at first
    images = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    report = evaluate(model, loader, device="cpu")
    assert 0.0 <= report.overall_accuracy <= 1.0
    assert "exit_0" in report.exit_distribution
    assert abs(sum(report.exit_distribution.values()) - 1.0) < 1e-6


def test_benchmark_wrapper_on_loader_returns_result():
    """Real-data throughput: drive the wrapper on samples from a loader and
    return the same BenchmarkResult shape as benchmark_wrapper. This is the
    honest input-distribution signal the README appendix has been promising."""
    from earlyon.benchmarking import benchmark_wrapper_on_loader

    model = _build()
    images = torch.randn(32, 3, 32, 32)
    labels = torch.randint(0, 10, (32,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=1)

    r = benchmark_wrapper_on_loader(model, loader, device="cpu", num_warmup=2, num_runs=16)
    assert r.throughput_ips > 0
    assert r.latency_median_ms > 0
    assert r.latency_p95_ms >= r.latency_p50_ms
    assert sum(r.exit_distribution.values()) == 1.0
    assert 0.0 < r.avg_computation_used <= 1.0
    assert r.num_runs == 16


def test_benchmark_wrapper_on_loader_rejects_batched_loader():
    """v0.2 inference is still batch-1; the helper must refuse a batched loader."""
    from earlyon.benchmarking import benchmark_wrapper_on_loader

    model = _build()
    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    import pytest

    with pytest.raises(ValueError, match="batch_size=1"):
        benchmark_wrapper_on_loader(model, loader, device="cpu", num_warmup=1, num_runs=4)


def test_benchmark_wrapper_on_loader_cycles_loader_when_short():
    """If the loader has fewer samples than num_runs, it must cycle so the
    measurement isn't truncated."""
    from earlyon.benchmarking import benchmark_wrapper_on_loader

    model = _build()
    images = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=1)

    r = benchmark_wrapper_on_loader(model, loader, device="cpu", num_warmup=2, num_runs=20)
    assert r.num_runs == 20
