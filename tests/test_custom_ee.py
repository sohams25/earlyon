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
    """Unknown layer names get the library's usual ValueError naming available
    layers (was: torch's raw AttributeError)."""
    with pytest.raises(ValueError, match="does_not_exist"):
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


def test_custom_ee_bad_layer_name_raises_value_error_listing_layers():
    """A typo'd exit layer must produce the library's usual clear ValueError
    (naming some available layers), not torch's raw AttributeError."""
    backbone = TinyBackbone(num_classes=10)
    with pytest.raises(ValueError, match="stage1"):
        custom_ee(backbone, ["stage_one_typo"], num_classes=10, input_shape=(1, 3, 32, 32))


# ---------------- v0.3 interface: examples, devices, adapters, validation ----------------


class _KwargBackbone(torch.nn.Module):
    """Backbone whose forward takes an extra kwarg with a runtime default."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.head = torch.nn.Linear(8, num_classes)

    def forward(self, x, scale: float = 1.0):
        f = self.stem(x) * scale
        return self.head(f.mean(dim=(2, 3)))


class _TupleOutputBackbone(torch.nn.Module):
    """The exit layer emits a (features, aux) tuple — needs an extractor."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.block = _TupleBlock()
        self.head = torch.nn.Linear(8, num_classes)

    def forward(self, x):
        f, _aux = self.block(x)
        return self.head(f.mean(dim=(2, 3)))


class _TupleBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        f = self.conv(x)
        return f, {"aux": f.sum()}


class _ReusedLayerBackbone(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.head = torch.nn.Linear(3, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv(x)  # same module twice
        return self.head(x.mean(dim=(2, 3)))


def test_custom_ee_accepts_example_args_and_kwargs():
    backbone = _KwargBackbone()
    model = custom_ee(
        backbone,
        exit_layers=["stem"],
        num_classes=10,
        input_shape=(1, 3, 16, 16),
        example_args=(torch.zeros(1, 3, 16, 16),),
        example_kwargs={"scale": 2.0},
    )
    model.eval()
    result = model(torch.randn(1, 3, 16, 16), mode="inference")
    assert result.prediction.shape == (1, 10)


def test_custom_ee_example_kwargs_without_args_rejected():
    with pytest.raises(ValueError, match="example_kwargs"):
        custom_ee(
            _KwargBackbone(),
            exit_layers=["stem"],
            num_classes=10,
            example_kwargs={"scale": 2.0},
        )


def test_custom_ee_tuple_output_needs_and_uses_feature_extractor():
    backbone = _TupleOutputBackbone()
    # without an extractor: clear error naming the fix
    with pytest.raises(ValueError, match="feature_extractors"):
        custom_ee(backbone, exit_layers=["block"], num_classes=10, input_shape=(1, 3, 16, 16))
    # with one: wraps and routes
    model = custom_ee(
        _TupleOutputBackbone(),
        exit_layers=["block"],
        num_classes=10,
        input_shape=(1, 3, 16, 16),
        feature_extractors={"block": lambda out: out[0]},
    )
    model.eval()
    model.config.confidence_thresholds = [0.0]  # force the exit to fire
    result = model(torch.randn(1, 3, 16, 16), mode="inference")
    assert result.exit_taken == 0  # the adapter ran inside the routing hook
    assert result.prediction.shape == (1, 10)


def test_custom_ee_rejects_extractor_for_unknown_layer():
    with pytest.raises(ValueError, match="unknown exit layer"):
        custom_ee(
            _TupleOutputBackbone(),
            exit_layers=["block"],
            num_classes=10,
            input_shape=(1, 3, 16, 16),
            feature_extractors={"block": lambda o: o[0], "nope": lambda o: o},
        )


def test_custom_ee_rejects_reused_exit_layer():
    with pytest.raises(RuntimeError, match="more than once"):
        custom_ee(
            _ReusedLayerBackbone(),
            exit_layers=["conv"],
            num_classes=10,
            input_shape=(1, 3, 16, 16),
        )


def test_custom_ee_rejects_out_of_order_exit_layers():
    backbone = TinyBackbone(num_classes=10)
    with pytest.raises(RuntimeError, match="forward-execution order"):
        custom_ee(
            backbone,
            exit_layers=["stage2", "stage1"],  # reversed
            num_classes=10,
            input_shape=(1, 3, 32, 32),
        )


def test_custom_ee_restores_training_mode_after_dry_run():
    backbone = TinyBackbone(num_classes=10)
    backbone.train()
    custom_ee(backbone, exit_layers=["stage1"], num_classes=10, input_shape=(1, 3, 32, 32))
    assert backbone.training is True


def test_custom_ee_infers_device_from_backbone():
    """A meta-level check that stays CPU-only: the dry-run tensor must follow
    the backbone's parameter device rather than defaulting to CPU."""
    from earlyon.models.custom import _infer_backbone_device

    backbone = TinyBackbone(num_classes=10)
    assert _infer_backbone_device(backbone).type == "cpu"


@pytest.mark.gpu
def test_custom_ee_wraps_cuda_backbone_without_device_error():
    backbone = TinyBackbone(num_classes=10).cuda()
    model = custom_ee(backbone, exit_layers=["stage1"], num_classes=10, input_shape=(1, 3, 32, 32))
    model.eval()
    result = model(torch.randn(1, 3, 32, 32, device="cuda"), mode="inference")
    assert result.prediction.shape == (1, 10)
