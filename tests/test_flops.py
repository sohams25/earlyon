"""Tests for FLOPs accounting, including the overcount warning."""

import warnings

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
