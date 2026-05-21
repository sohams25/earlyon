"""Wrapper invariants and routing behavior."""

import pytest
import torch

from earlyon.core.exit_head import EarlyExitHead
from earlyon.core.types import EarlyExitConfig, ExitPoint
from earlyon.core.wrappers import EarlyExitWrapper
from tests.fixtures.tiny_models import STAGE_CHANNELS, TinyBackbone


def _build(thresholds=(0.99, 0.99, 0.99)):
    backbone = TinyBackbone(num_classes=10)
    exits = [
        ExitPoint("e0", "stage1", STAGE_CHANNELS["stage1"]),
        ExitPoint("e1", "stage2", STAGE_CHANNELS["stage2"]),
        ExitPoint("e2", "stage3", STAGE_CHANNELS["stage3"]),
    ]
    heads = {ep.name: EarlyExitHead(ep.in_channels, 10) for ep in exits}
    cfg = EarlyExitConfig(
        backbone="tiny",
        num_classes=10,
        exit_points=exits,
        confidence_thresholds=list(thresholds),
    )

    def final_classifier(feats):
        # feats is already the backbone output (logits) since TinyBackbone.fc
        # runs internally. We pass through identity.
        return feats

    wrapper = EarlyExitWrapper(
        backbone=backbone,
        exit_heads=heads,
        final_classifier=final_classifier,
        config=cfg,
        input_shape=(1, 3, 32, 32),
    )
    return wrapper, backbone


def test_training_mode_returns_all_exits():
    wrapper, _ = _build()
    x = torch.randn(4, 3, 32, 32)
    outputs = wrapper(x, mode="training")
    assert len(outputs) == 4  # 3 exits + final
    for o in outputs:
        assert o.shape == (4, 10)


def test_inference_returns_inference_result():
    wrapper, _ = _build()
    x = torch.randn(1, 3, 32, 32)
    result = wrapper(x, mode="inference")
    assert result.prediction.shape == (1, 10)
    assert -1 <= result.exit_taken < 3
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 < result.computation_used <= 1.0


def test_thresholds_one_means_no_early_exit():
    """The core invariant: with thresholds=1.0, no exit can fire because
    softmax(logits).max() is strictly < 1. The wrapper must reach the final
    classifier and return the backbone output unchanged."""
    wrapper, backbone = _build(thresholds=(1.0, 1.0, 1.0))
    wrapper.eval()
    backbone.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        result = wrapper(x, mode="inference")
        direct = backbone(x)
    assert result.exit_taken == -1
    assert result.computation_used == 1.0
    assert torch.allclose(result.prediction, direct, atol=1e-5)


def test_zero_threshold_exits_immediately():
    """With threshold=0, the first exit always fires."""
    wrapper, _ = _build(thresholds=(0.0, 0.0, 0.0))
    wrapper.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        result = wrapper(x, mode="inference")
    assert result.exit_taken == 0
    assert result.computation_used < 1.0


def test_inference_rejects_batch_gt_1():
    wrapper, _ = _build(thresholds=(0.0, 0.0, 0.0))
    x = torch.randn(2, 3, 32, 32)
    with pytest.raises(RuntimeError, match="batch_size=1"):
        wrapper(x, mode="inference")


def test_exit_parameters_excludes_backbone():
    wrapper, _ = _build()
    exit_param_ids = {id(p) for p in wrapper.exit_parameters()}
    backbone_param_ids = {id(p) for p in wrapper.backbone_parameters()}
    assert exit_param_ids.isdisjoint(backbone_param_ids)
    assert len(exit_param_ids) > 0


def test_computation_used_monotonic_across_exits():
    """Later exits must report higher cumulative FLOPs than earlier ones."""
    wrapper, _ = _build(thresholds=(0.0, 0.0, 0.0))
    wrapper.eval()
    flops_at = wrapper._flops_at
    values = [flops_at[ep.layer_name] for ep in wrapper.config.exit_points]
    assert values == sorted(values)
    assert all(0 < v <= 1 for v in values)
