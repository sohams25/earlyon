"""Integration tests for vit_b_16_ee (the ViT-B/16 early-exit factory).

Building vit_b_16 is ~86M params; a module-scoped fixture builds it once
(pretrained=False, no download) and shares it. Tests set their own thresholds,
so the shared instance is safe to reuse.
"""

import pytest
import torch

from earlyon.models import vit_b_16_ee


@pytest.fixture(scope="module")
def vit():
    return vit_b_16_ee(num_classes=10, pretrained=False)


def test_factory_default_wiring(vit):
    """Pin the factory's default config so a future edit can't silently change it."""
    assert vit.config.confidence_thresholds == [0.85, 0.80]
    assert vit.config.loss_weights == [0.2, 0.3, 0.5]
    assert vit.config.backbone == "vit_b_16"


def test_exit_points_placed_at_blocks_3_and_9(vit):
    points = vit.config.exit_points
    assert [ep.layer_name for ep in points] == [
        "encoder.layers.encoder_layer_3",
        "encoder.layers.encoder_layer_9",
    ]
    assert [ep.in_channels for ep in points] == [768, 768]


def test_flops_fractions_count_attention_not_just_leaves(vit):
    """Regression for the MHA-undercount bug: fvcore puts each block's attention
    FLOPs on the non-leaf self_attention module, so a leaf-only walk reported
    ~0.23/0.56. The corrected accounting must report ~0.34/0.83."""
    values = [vit._flops_at[ep.layer_name] for ep in vit.config.exit_points]
    assert values == sorted(values)
    assert all(0.0 < v < 1.0 for v in values)
    assert values[0] == pytest.approx(0.338, abs=0.02), f"block-3 fraction {values[0]}"
    assert values[1] == pytest.approx(0.834, abs=0.02), f"block-9 fraction {values[1]}"


def test_training_mode_returns_all_outputs(vit):
    vit.eval()
    with torch.no_grad():
        outputs = vit(torch.randn(1, 3, 224, 224), mode="training")
    assert len(outputs) == 3  # 2 exits + final
    assert all(o.shape == (1, 10) for o in outputs)


def test_zero_threshold_exits_at_first_block(vit):
    vit.config.confidence_thresholds = [0.0, 0.0]
    vit.eval()
    result = vit(torch.randn(1, 3, 224, 224), mode="inference")
    assert result.exit_taken == 0
    assert result.computation_used < 1.0


def test_high_threshold_reaches_final_and_matches_backbone(vit):
    vit.config.confidence_thresholds = [1.0, 1.0]
    vit.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        result = vit(x, mode="inference")
        direct = vit.backbone(x)
    assert result.exit_taken == -1
    assert torch.allclose(result.prediction, direct, atol=1e-5)


def test_inference_prediction_is_grad_free(vit):
    vit.config.confidence_thresholds = [0.0, 0.0]
    vit.eval()
    result = vit(torch.randn(1, 3, 224, 224), mode="inference")
    assert result.prediction.requires_grad is False
    assert result.prediction.grad_fn is None


def test_num_classes_replaces_head():
    """num_classes != 1000 swaps the ViT head; a fresh build (not the fixture)."""
    model = vit_b_16_ee(num_classes=100, pretrained=False)
    assert model.backbone.heads.head.out_features == 100
    model.eval()
    with torch.no_grad():
        result = model(torch.randn(1, 3, 224, 224), mode="inference")
    assert result.prediction.shape == (1, 100)
