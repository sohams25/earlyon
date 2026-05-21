import pytest
import torch

from earlyon.models import mobilenetv2_ee, resnet18_ee, resnet50_ee


@pytest.mark.parametrize(
    "factory,n_exits,input_shape",
    [
        (lambda: resnet18_ee(num_classes=10, pretrained=False), 2, (1, 3, 224, 224)),
        (lambda: resnet50_ee(num_classes=10, pretrained=False), 3, (1, 3, 224, 224)),
        (lambda: mobilenetv2_ee(num_classes=10, pretrained=False), 2, (1, 3, 224, 224)),
    ],
    ids=["resnet18", "resnet50", "mobilenetv2"],
)
def test_factory_builds_and_routes(factory, n_exits, input_shape):
    model = factory()
    model.eval()
    assert len(model.config.exit_points) == n_exits

    # training mode returns n_exits + 1 outputs
    x = torch.randn(*input_shape)
    with torch.no_grad():
        outs = model(x, mode="training")
    assert len(outs) == n_exits + 1
    for o in outs:
        assert o.shape == (1, 10)

    # inference mode returns InferenceResult
    with torch.no_grad():
        result = model(x, mode="inference")
    assert result.prediction.shape == (1, 10)


def test_resnet50_identity_invariant():
    """thresholds=1.0 => wrapper output must equal backbone output."""
    model = resnet50_ee(num_classes=10, pretrained=False)
    model.config.confidence_thresholds = [1.0, 1.0, 1.0]
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        result = model(x, mode="inference")
        direct = model.backbone(x)
    assert result.exit_taken == -1
    assert torch.allclose(result.prediction, direct, atol=1e-5)


def test_mobilenetv2_identity_invariant():
    model = mobilenetv2_ee(num_classes=10, pretrained=False)
    model.config.confidence_thresholds = [1.0, 1.0]
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        result = model(x, mode="inference")
        direct = model.backbone(x)
    assert result.exit_taken == -1
    assert torch.allclose(result.prediction, direct, atol=1e-5)


def test_efficientnet_b0_builds_and_routes():
    from earlyon.models import efficientnet_b0_ee
    model = efficientnet_b0_ee(num_classes=10, pretrained=False)
    model.eval()
    assert len(model.config.exit_points) == 2

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        outs = model(x, mode="training")
    assert len(outs) == 3

    with torch.no_grad():
        result = model(x, mode="inference")
    assert result.prediction.shape == (1, 10)


def test_efficientnet_b0_identity_invariant():
    from earlyon.models import efficientnet_b0_ee
    model = efficientnet_b0_ee(num_classes=10, pretrained=False)
    model.config.confidence_thresholds = [1.0, 1.0]
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        result = model(x, mode="inference")
        direct = model.backbone(x)
    assert result.exit_taken == -1
    assert torch.allclose(result.prediction, direct, atol=1e-4)
