"""Jetson profiler tests. These run on a non-Jetson host: tegrastats is
absent so power/temp fields are None (missing, not invented), but the
harness must still execute end-to-end. Parser, lifecycle, and energy
integration are tested without any hardware."""

import time

import pytest

from earlyon.benchmarking.jetson_profiler import (
    JetsonProfiler,
    JetsonSample,
    TegrastatsMonitor,
    _parse_line,
    integrate_energy,
)
from earlyon.core.exit_head import EarlyExitHead
from earlyon.core.types import EarlyExitConfig, ExitPoint
from earlyon.core.wrappers import EarlyExitWrapper
from tests.fixtures.tiny_models import STAGE_CHANNELS, TinyBackbone

SAMPLE_LINE = (
    "RAM 3456/7772MB SWAP 0/3886MB CPU [12%@1200,8%@1200,5%@1200,3%@1200] "
    "GR3D_FREQ 75% GR3D2_FREQ 0% tj@42.5C VDD_IN 4500 VDD_CPU_GPU_CV 1200"
)

# Orin-style format variant: different power rail name, PLL temp
ORIN_LINE = "RAM 2000/16000MB GR3D_FREQ 12% pll@38C VDD_GPU 800"


def _wrapper():
    backbone = TinyBackbone(num_classes=10)
    exits = [ExitPoint("e0", "stage1", STAGE_CHANNELS["stage1"])]
    heads = {ep.name: EarlyExitHead(ep.in_channels, 10) for ep in exits}
    cfg = EarlyExitConfig(
        backbone="tiny",
        num_classes=10,
        exit_points=exits,
        confidence_thresholds=[0.8],
    )
    return EarlyExitWrapper(backbone, heads, lambda x: x, cfg, input_shape=(1, 3, 32, 32))


def test_parse_line_extracts_gpu_temp_power():
    s = _parse_line(SAMPLE_LINE)
    assert s.gpu_util_pct == 75.0
    assert s.temp_c == 42.5
    assert s.power_mw == 4500.0


def test_parse_line_orin_variant():
    s = _parse_line(ORIN_LINE)
    assert s.gpu_util_pct == 12.0
    assert s.temp_c == 38.0
    assert s.power_mw == 800.0


def test_parse_line_missing_fields_are_none_not_zero():
    """A line without a field must report None — 0 mW is a real (wrong)
    measurement, None is honestly missing."""
    s = _parse_line("nothing useful here")
    assert s.gpu_util_pct is None
    assert s.temp_c is None
    assert s.power_mw is None


def test_monitor_safe_to_start_stop_without_tegrastats():
    """On a host without tegrastats the monitor must not crash."""
    m = TegrastatsMonitor()
    m.start()
    sample = m.latest()
    assert sample.timestamp > 0
    m.stop()


def test_monitor_is_restartable():
    """Regression: stop() set a threading.Event that start() never cleared, so
    a restarted monitor's reader thread exited immediately and silently
    collected nothing. A stop/start cycle must yield a working session."""
    m = TegrastatsMonitor()
    m.start()
    m.stop()
    assert not m._stop.is_set() or not m.available  # unavailable hosts no-op
    m.start()
    if m.available:
        assert m.running
        assert not m._stop.is_set()
    m.stop()
    assert m._proc is None and m._thread is None


def test_monitor_double_start_is_noop_not_duplicate():
    m = TegrastatsMonitor()
    m.start()
    proc = m._proc
    m.start()  # must not spawn a second process/thread
    assert m._proc is proc
    m.stop()


def test_monitor_validates_arguments():
    with pytest.raises(ValueError, match="interval_ms"):
        TegrastatsMonitor(interval_ms=0)
    with pytest.raises(ValueError, match="max_samples"):
        TegrastatsMonitor(max_samples=0)


def test_profiler_runs_on_cpu_without_tegrastats():
    profiler = JetsonProfiler()
    runs = profiler.profile(
        _wrapper(), input_shape=(1, 3, 32, 32), num_warmup=2, num_runs=5, device="cpu"
    )
    assert len(runs) == 5
    for r in runs:
        assert r.latency_ms > 0
        # no tegrastats on this host -> telemetry is missing, not zero
        if not profiler.monitor.available:
            assert r.gpu_util_pct is None
            assert r.power_mw is None


def test_profiler_reports_energy_summary_without_fabrication():
    profiler = JetsonProfiler()
    runs, summary = profiler.profile_with_energy(
        _wrapper(), input_shape=(1, 3, 32, 32), num_warmup=1, num_runs=4, device="cpu"
    )
    assert len(runs) == 4
    assert summary.num_inferences == 4
    assert summary.window_seconds > 0
    if not profiler.monitor.available:
        assert summary.num_power_samples == 0
        assert summary.avg_power_mw is None
        assert summary.energy_mj is None
        assert summary.energy_per_inference_mj is None


def test_profiler_validates_arguments():
    profiler = JetsonProfiler()
    with pytest.raises(ValueError, match="num_runs"):
        profiler.profile(_wrapper(), input_shape=(1, 3, 32, 32), num_runs=0, device="cpu")
    with pytest.raises(ValueError, match="num_warmup"):
        profiler.profile(_wrapper(), input_shape=(1, 3, 32, 32), num_warmup=-1, device="cpu")


# ---------------- energy integration (pure, no hardware) ----------------


def _sample(t: float, mw: float | None) -> JetsonSample:
    return JetsonSample(timestamp=t, gpu_util_pct=None, temp_c=None, power_mw=mw)


def test_integrate_energy_trapezoid():
    # constant 1000 mW over 2 s -> 2000 mJ; 10 inferences -> 200 mJ each
    samples = [_sample(10.0, 1000.0), _sample(11.0, 1000.0), _sample(12.0, 1000.0)]
    s = integrate_energy(samples, 10.0, 12.0, num_inferences=10, sampling_interval_ms=1000)
    assert s.num_power_samples == 3
    assert s.avg_power_mw == 1000.0
    assert s.energy_mj == pytest.approx(2000.0)
    assert s.energy_per_inference_mj == pytest.approx(200.0)
    assert s.window_seconds == pytest.approx(2.0)


def test_integrate_energy_single_sample_yields_no_energy():
    """One instantaneous sample cannot be integrated; avg power is reported,
    energy is honestly None."""
    s = integrate_energy([_sample(10.0, 500.0)], 9.0, 11.0, 5, 100)
    assert s.avg_power_mw == 500.0
    assert s.energy_mj is None
    assert s.energy_per_inference_mj is None


def test_integrate_energy_ignores_samples_outside_window_and_missing_power():
    samples = [
        _sample(5.0, 9999.0),  # before window
        _sample(10.0, 1000.0),
        _sample(10.5, None),  # power missing on this line
        _sample(11.0, 1000.0),
        _sample(20.0, 9999.0),  # after window
    ]
    s = integrate_energy(samples, 10.0, 11.0, 2, 500)
    assert s.num_power_samples == 2
    assert s.energy_mj == pytest.approx(1000.0)  # 1000 mW * 1 s
    assert s.energy_per_inference_mj == pytest.approx(500.0)


def test_integrate_energy_no_samples():
    s = integrate_energy([], time.monotonic(), time.monotonic() + 1, 3, 100)
    assert s.num_power_samples == 0
    assert s.avg_power_mw is None and s.energy_mj is None
