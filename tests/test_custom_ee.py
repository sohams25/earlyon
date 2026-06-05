"""Tests for custom_ee — wrapping arbitrary backbones with auto-inferred exits."""

import pytest
import torch
import torch.nn as nn

from earlyon.models import custom_ee
from tests.fixtures.tiny_models import STAGE_CHANNELS, TinyBackbone, TinyTokenBackbone


def test_custom_ee_infers_conv_channels():
    """4D conv features: in_channels auto-inferred from the channel dim."""
    model = custom_ee(
        TinyBackbone(num_classes=10),
        ["stage1", "stage3"],
        num_classes=10,
        input_shape=(1, 3, 32, 32),
    )
    widths = [ep.in_channels for ep in model.config.exit_points]
    assert widths == [STAGE_CHANNELS["stage1"], STAGE_CHANNELS["stage3"]]


def test_custom_ee_infers_token_channels():
    """3D token features: in_channels auto-inferred from the embedding dim."""
    model = custom_ee(
        TinyTokenBackbone(num_classes=10, dim=32),
        ["block0", "block1"],
        num_classes=10,
        input_shape=(1, 3, 32, 32),
    )
    assert [ep.in_channels for ep in model.config.exit_points] == [32, 32]


def test_custom_ee_conv_training_mode_returns_all_outputs():
    model = custom_ee(
        TinyBackbone(num_classes=10),
        ["stage1", "stage2"],
        num_classes=10,
        input_shape=(1, 3, 32, 32),
    )
    outputs = model(torch.randn(4, 3, 32, 32), mode="training")
    assert len(outputs) == 3  # 2 exits + final
    assert all(o.shape == (4, 10) for o in outputs)


def test_custom_ee_token_inference_routes():
    model = custom_ee(
        TinyTokenBackbone(num_classes=10),
        ["block0", "block1"],
        num_classes=10,
        input_shape=(1, 3, 32, 32),
        confidence_thresholds=[0.0, 0.0],  # always exit e0
    )
    model.eval()
    result = model(torch.randn(1, 3, 32, 32), mode="inference")
    assert result.exit_taken == 0
    assert result.computation_used < 1.0


def test_custom_ee_identity_invariant_no_exit():
    """thresholds=1.0 -> no exit -> prediction equals the bare backbone output."""
    backbone = TinyBackbone(num_classes=10)
    model = custom_ee(
        backbone,
        ["stage1"],
        num_classes=10,
        input_shape=(1, 3, 32, 32),
        confidence_thresholds=[1.0],
    )
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        result = model(x, mode="inference")
        direct = backbone(x)
    assert result.exit_taken == -1
    assert torch.allclose(result.prediction, direct, atol=1e-5)


@pytest.mark.parametrize("start_training", [True, False])
def test_custom_ee_preserves_backbone_training_mode(start_training):
    """The dry-run must restore the backbone's prior mode — both branches."""
    backbone = TinyBackbone(num_classes=10)
    backbone.train(start_training)
    custom_ee(backbone, ["stage1"], num_classes=10, input_shape=(1, 3, 32, 32))
    assert backbone.training is start_training


def test_custom_ee_entropy_routing():
    """The entropy routing_policy is wired through to the config and routes."""
    import math

    model = custom_ee(
        TinyTokenBackbone(num_classes=10),
        ["block0", "block1"],
        num_classes=10,
        input_shape=(1, 3, 32, 32),
        routing_policy="entropy",
    )
    assert model.config.routing_policy == "entropy"
    assert len(model.config.entropy_thresholds) == 2
    # wide entropy threshold -> exit immediately
    model.config.entropy_thresholds = [math.log(10) + 0.1] * 2
    model.eval()
    with torch.no_grad():
        result = model(torch.randn(1, 3, 32, 32), mode="inference")
    assert result.exit_taken == 0


def test_custom_ee_rejects_empty_exit_layers():
    with pytest.raises(ValueError, match="non-empty"):
        custom_ee(TinyBackbone(num_classes=10), [], num_classes=10, input_shape=(1, 3, 32, 32))


def test_custom_ee_rejects_unknown_layer_name():
    with pytest.raises(AttributeError):
        custom_ee(
            TinyBackbone(num_classes=10),
            ["does_not_exist"],
            num_classes=10,
            input_shape=(1, 3, 32, 32),
        )


def test_custom_ee_rejects_non_tensor_layer_output():
    """A layer that returns a non-Tensor can't have its width inferred."""

    class TupleBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weird = nn.Identity()
            self.fc = nn.Linear(3 * 8 * 8, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _ = self.weird((x, x))  # hook sees a tuple output
            return self.fc(x.flatten(1))

    with pytest.raises(ValueError, match="not a Tensor"):
        custom_ee(TupleBackbone(), ["weird"], num_classes=10, input_shape=(1, 3, 8, 8))
