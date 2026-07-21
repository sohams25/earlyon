"""Tests for FLOPs accounting, including the overcount warning."""

import sys
import warnings

import pytest
import torch.nn as nn

from earlyon.core.flops import per_layer_flops


class _FakeFCA:
    """Stand-in for fvcore's FlopCountAnalysis that lets a test force a
    contrived ratio between sum-of-leaves and total."""

    def __init__(self, total: int, by_module: dict[str, int]):
        self._total = total
        self._by_module = by_module

    def total(self):
        return self._total

    def by_module(self):
        return self._by_module

    def unsupported_ops_warnings(self, _):
        return self

    def uncalled_modules_warnings(self, _):
        return self


def _toy_backbone():
    return nn.Sequential(
        nn.Conv2d(3, 8, 3),
        nn.Conv2d(8, 8, 3),
    )


def test_overcount_emits_warning(monkeypatch):
    backbone = _toy_backbone()

    # Force ratio of 2.0: total=10, both leaves report 10, sum=20.
    fake = _FakeFCA(total=10, by_module={"0": 10, "1": 10})
    import fvcore.nn

    monkeypatch.setattr(fvcore.nn, "FlopCountAnalysis", lambda *a, **k: fake)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = per_layer_flops(backbone, ["1"], input_shape=(1, 3, 8, 8))

    assert any(
        issubclass(w.category, RuntimeWarning) and "overcount" in str(w.message) for w in caught
    ), f"expected RuntimeWarning about overcount, got: {[str(w.message) for w in caught]}"
    # Clamped to 1.0 despite the overcount
    assert result["1"] == 1.0


def test_no_warning_when_ratio_below_threshold(monkeypatch):
    backbone = _toy_backbone()
    fake = _FakeFCA(total=10, by_module={"0": 5, "1": 5})
    import fvcore.nn

    monkeypatch.setattr(fvcore.nn, "FlopCountAnalysis", lambda *a, **k: fake)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = per_layer_flops(backbone, ["1"], input_shape=(1, 3, 8, 8))

    overcount_warnings = [w for w in caught if "overcount" in str(w.message)]
    assert overcount_warnings == []
    assert result["1"] == 1.0


def test_per_layer_flops_restores_training_mode():
    """The FLOPs probe eval()s the backbone but must restore the prior mode —
    otherwise constructing a wrapper from a training backbone silently freezes
    BatchNorm in a custom training loop."""
    backbone = _toy_backbone()
    backbone.train()
    per_layer_flops(backbone, ["1"], input_shape=(1, 3, 8, 8))
    assert backbone.training is True  # restored, not left in eval


def test_uniform_fallback_when_fvcore_missing(monkeypatch):
    """When fvcore can't be imported, per_layer_flops falls back to uniform
    spacing rather than crashing — the library must work without fvcore."""
    monkeypatch.setitem(sys.modules, "fvcore.nn", None)  # makes the import raise
    backbone = _toy_backbone()
    result = per_layer_flops(backbone, ["0", "1"], input_shape=(1, 3, 8, 8))
    assert result["0"] == pytest.approx(1 / 3)
    assert result["1"] == pytest.approx(2 / 3)


def test_uniform_fallback_when_total_flops_zero(monkeypatch):
    """A degenerate backbone reporting zero total FLOPs must not divide by zero;
    it falls back to uniform spacing."""
    import fvcore.nn

    fake = _FakeFCA(total=0, by_module={})
    monkeypatch.setattr(fvcore.nn, "FlopCountAnalysis", lambda *a, **k: fake)
    backbone = _toy_backbone()
    result = per_layer_flops(backbone, ["0", "1"], input_shape=(1, 3, 8, 8))
    assert result["0"] == pytest.approx(1 / 3)
    assert result["1"] == pytest.approx(2 / 3)


def test_uniform_fallback_for_layer_name_with_no_leaves(monkeypatch):
    """A layer_name that matches no leaf module falls back to its index-based
    uniform value instead of silently dropping the exit."""
    import fvcore.nn

    fake = _FakeFCA(total=10, by_module={"0": 5, "1": 5})
    monkeypatch.setattr(fvcore.nn, "FlopCountAnalysis", lambda *a, **k: fake)
    backbone = _toy_backbone()
    result = per_layer_flops(backbone, ["nonexistent"], input_shape=(1, 3, 8, 8))
    assert result["nonexistent"] == pytest.approx(1 / 2)


# ---------------- estimator metadata + reuse detection (v0.3) ----------------


def test_estimate_layer_flops_reports_method_and_reliability():
    from earlyon.core.flops import METHOD_FVCORE, estimate_layer_flops

    backbone = _toy_backbone()
    est = estimate_layer_flops(backbone, ["0", "1"], input_shape=(1, 3, 8, 8))
    assert est.method == METHOD_FVCORE
    assert est.reliable is True
    assert est.excludes_exit_heads is True
    assert set(est.fractions) == {"0", "1"}
    assert est.fractions["0"] <= est.fractions["1"] <= 1.0


def test_reused_module_falls_back_to_low_confidence_uniform():
    """A backbone that calls the same leaf twice breaks the module-order
    assumption; the estimator must warn and return reliable=False instead of a
    precise-looking but wrong fraction."""

    from earlyon.core.flops import METHOD_UNIFORM, estimate_layer_flops

    class Reuser(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 3, padding=1)
            self.head = nn.Conv2d(3, 8, 3, padding=1)

        def forward(self, x):
            x = self.conv(x)
            x = self.conv(x)  # same module twice
            return self.head(x)

    backbone = Reuser()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est = estimate_layer_flops(backbone, ["conv"], input_shape=(1, 3, 8, 8))
    assert est.method == METHOD_UNIFORM
    assert est.reliable is False
    assert any("reuses module" in str(w.message) for w in caught)


def test_wrapper_flops_analysis_is_lazy():
    """Constructing a wrapper must NOT run the FLOPs analysis (it made ViT
    construction unacceptably slow); the first inference materialises it."""
    from unittest.mock import patch

    import torch

    from earlyon.core.exit_head import EarlyExitHead
    from earlyon.core.types import EarlyExitConfig, ExitPoint
    from earlyon.core.wrappers import EarlyExitWrapper
    from tests.fixtures.tiny_models import STAGE_CHANNELS, TinyBackbone

    with patch("earlyon.core.wrappers.estimate_layer_flops", autospec=True) as mocked:
        from earlyon.core.flops import estimate_layer_flops as real

        mocked.side_effect = real
        backbone = TinyBackbone(num_classes=10)
        exits = [ExitPoint("e0", "stage1", STAGE_CHANNELS["stage1"])]
        heads = {"e0": EarlyExitHead(STAGE_CHANNELS["stage1"], 10)}
        cfg = EarlyExitConfig(backbone="tiny", num_classes=10, exit_points=exits)
        wrapper = EarlyExitWrapper(backbone, heads, lambda t: t, cfg, input_shape=(1, 3, 32, 32))
        assert mocked.call_count == 0, "construction must not run the FLOPs analysis"

        wrapper.eval()
        wrapper(torch.randn(1, 3, 32, 32), mode="inference")
        assert mocked.call_count == 1

        # cached: further inferences do not re-run the analysis
        wrapper(torch.randn(1, 3, 32, 32), mode="inference")
        assert mocked.call_count == 1


def test_wrapper_exposes_flops_estimate_metadata():

    from earlyon.core.exit_head import EarlyExitHead
    from earlyon.core.types import EarlyExitConfig, ExitPoint
    from earlyon.core.wrappers import EarlyExitWrapper
    from tests.fixtures.tiny_models import STAGE_CHANNELS, TinyBackbone

    backbone = TinyBackbone(num_classes=10)
    exits = [ExitPoint("e0", "stage1", STAGE_CHANNELS["stage1"])]
    heads = {"e0": EarlyExitHead(STAGE_CHANNELS["stage1"], 10)}
    cfg = EarlyExitConfig(backbone="tiny", num_classes=10, exit_points=exits)
    wrapper = EarlyExitWrapper(backbone, heads, lambda t: t, cfg, input_shape=(1, 3, 32, 32))
    est = wrapper.flops_estimate
    assert est.excludes_exit_heads is True
    assert 0.0 < est.fractions["stage1"] <= 1.0
