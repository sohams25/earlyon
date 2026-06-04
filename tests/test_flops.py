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
