"""Staged-deployment contract: staged execution must match eager routing."""

import pytest
import torch
import torch.nn as nn

from earlyon.models import custom_ee
from earlyon.staged import StagedModel, staged_model
from tests.fixtures.tiny_models import TinyBackbone


def _sequential_backbone(num_classes: int = 10) -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, num_classes),
    )


def _wrapped(**kwargs):
    backbone = _sequential_backbone()
    return custom_ee(
        backbone,
        exit_layers=["0", "2"],
        num_classes=10,
        input_shape=(1, 3, 16, 16),
        **kwargs,
    )


def test_staged_matches_eager_across_configs():
    torch.manual_seed(1)
    wrapper = _wrapped()
    staged = staged_model(wrapper)
    wrapper.eval()
    staged.eval()

    configs = [
        dict(thr=[0.99, 0.99], enabled=[True, True], temps=(1.0, 1.0, 1.0)),
        dict(thr=[0.0, 0.0], enabled=[True, True], temps=(1.0, 1.0, 1.0)),  # always exit 0
        dict(thr=[0.0, 0.0], enabled=[False, True], temps=(1.0, 1.0, 1.0)),  # exit 0 off
        dict(thr=[0.0, 0.0], enabled=[False, False], temps=(1.0, 1.0, 1.0)),  # all off
        dict(thr=[0.5, 0.5], enabled=[True, True], temps=(3.0, 0.5, 1.2)),  # per-head temps
    ]
    for cfg in configs:
        wrapper.config.confidence_thresholds = list(cfg["thr"])
        wrapper.config.enabled_exits = list(cfg["enabled"])
        t0, t1, tf = cfg["temps"]
        wrapper.config.temperatures = {"e0": t0, "e1": t1, "final": tf}
        for i in range(5):
            x = torch.randn(1, 3, 16, 16, generator=torch.Generator().manual_seed(i))
            eager = wrapper(x, mode="inference")
            staged_r = staged.infer(x)
            assert staged_r.exit_taken == eager.exit_taken, cfg
            assert torch.allclose(staged_r.prediction, eager.prediction, atol=1e-6), cfg
            assert staged_r.estimated_backbone_flops_fraction == pytest.approx(
                eager.estimated_backbone_flops_fraction
            )


def test_staged_matches_eager_entropy_policy():
    wrapper = _wrapped(routing_policy="entropy")
    wrapper.config.entropy_thresholds = [1.5, 1.5]
    staged = staged_model(wrapper)
    wrapper.eval()
    staged.eval()
    for i in range(5):
        x = torch.randn(1, 3, 16, 16, generator=torch.Generator().manual_seed(100 + i))
        eager = wrapper(x, mode="inference")
        staged_r = staged.infer(x)
        assert staged_r.exit_taken == eager.exit_taken
        assert torch.allclose(staged_r.prediction, eager.prediction, atol=1e-6)


def test_staged_stage_specs_partition_the_backbone():
    wrapper = _wrapped()
    staged = staged_model(wrapper)
    all_modules = [m for spec in staged.specs for m in spec.modules]
    assert all_modules == ["0", "1", "2", "3", "4", "5", "6"]  # full cover, in order
    assert staged.specs[0].exit_name == "e0"
    assert staged.specs[-1].exit_name == "final"


def test_staged_rejects_non_sequential_backbone():
    model = custom_ee(
        TinyBackbone(num_classes=10),
        exit_layers=["stage1"],
        num_classes=10,
        input_shape=(1, 3, 32, 32),
    )
    with pytest.raises(ValueError, match="nn.Sequential"):
        staged_model(model)


def test_staged_rejects_batched_input():
    staged = staged_model(_wrapped())
    with pytest.raises(ValueError, match="batch-1"):
        staged.infer(torch.randn(2, 3, 16, 16))


def test_staged_model_validates_stage_count():
    wrapper = _wrapped()
    good = staged_model(wrapper)
    with pytest.raises(ValueError, match="stage"):
        StagedModel(list(good.stages)[:-1], good.specs[:-1], wrapper.config, wrapper._flops_at)
