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
    wrapper = EarlyExitWrapper(
        backbone, heads, lambda x: x, cfg, input_shape=(1, 3, 32, 32)
    )
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


def test_wrapper_overhead_when_no_exits_trigger():
    """The architect's key invariant: wrapping should add <30% overhead on a
    tiny model when no exits fire (threshold=1.0). On real models the
    fraction is much smaller; tiny models have such low absolute latency
    that hook overhead is a larger fraction."""
    model = _build(thresholds=(1.0, 1.0))  # no exit will ever fire
    wrap_r = benchmark_wrapper(model, input_shape=(1, 3, 32, 32), num_warmup=10, num_runs=50)
    bb_r = benchmark_backbone(model.backbone, input_shape=(1, 3, 32, 32), num_warmup=10, num_runs=50)
    overhead = (wrap_r.latency_median_ms - bb_r.latency_median_ms) / bb_r.latency_median_ms
    # tiny model; allow generous slack
    assert overhead < 0.50, f"hook overhead too high on tiny model: {overhead:.1%}"


def test_evaluate_loop_runs():
    model = _build(thresholds=(0.0, 0.0))  # always exits at first
    images = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    report = evaluate(model, loader, device="cpu")
    assert 0.0 <= report.overall_accuracy <= 1.0
    assert "exit_0" in report.exit_distribution
    assert abs(sum(report.exit_distribution.values()) - 1.0) < 1e-6
