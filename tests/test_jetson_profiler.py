"""Jetson profiler tests. These run on a non-Jetson host: tegrastats is
absent so power/temp fields will be zero, but the harness must still execute
end-to-end."""

from earlyon.benchmarking.jetson_profiler import (
    JetsonProfiler,
    TegrastatsMonitor,
    _parse_line,
)
from earlyon.core.exit_head import EarlyExitHead
from earlyon.core.types import EarlyExitConfig, ExitPoint
from earlyon.core.wrappers import EarlyExitWrapper
from tests.fixtures.tiny_models import STAGE_CHANNELS, TinyBackbone

SAMPLE_LINE = (
    "RAM 3456/7772MB SWAP 0/3886MB CPU [12%@1200,8%@1200,5%@1200,3%@1200] "
    "GR3D_FREQ 75% GR3D2_FREQ 0% tj@42.5C VDD_IN 4500 VDD_CPU_GPU_CV 1200"
)


def test_parse_line_extracts_gpu_temp_power():
    s = _parse_line(SAMPLE_LINE)
    assert s.gpu_util_pct == 75.0
    assert s.temp_c == 42.5
    assert s.power_mw == 4500.0


def test_parse_line_missing_fields_returns_zero():
    s = _parse_line("nothing useful here")
    assert s.gpu_util_pct == 0.0
    assert s.temp_c == 0.0
    assert s.power_mw == 0.0


def test_monitor_safe_to_start_stop_without_tegrastats():
    """On a host without tegrastats the monitor must not crash."""
    m = TegrastatsMonitor()
    # We don't know the host; just ensure the lifecycle works regardless.
    m.start()
    sample = m.latest()
    assert sample.gpu_util_pct >= 0.0
    m.stop()


def test_profiler_runs_on_cpu_without_tegrastats():
    backbone = TinyBackbone(num_classes=10)
    exits = [ExitPoint("e0", "stage1", STAGE_CHANNELS["stage1"])]
    heads = {ep.name: EarlyExitHead(ep.in_channels, 10) for ep in exits}
    cfg = EarlyExitConfig(
        backbone="tiny",
        num_classes=10,
        exit_points=exits,
        confidence_thresholds=[0.8],
    )
    wrapper = EarlyExitWrapper(backbone, heads, lambda x: x, cfg, input_shape=(1, 3, 32, 32))
    profiler = JetsonProfiler()
    runs = profiler.profile(
        wrapper, input_shape=(1, 3, 32, 32), num_warmup=2, num_runs=5, device="cpu"
    )
    assert len(runs) == 5
    for r in runs:
        assert r.latency_ms > 0
        assert r.gpu_util_pct >= 0.0
        assert r.temp_c >= 0.0
        assert r.power_mw >= 0.0
