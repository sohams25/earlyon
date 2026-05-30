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


def test_cifar_resnet_ee_builds_for_32x32_input():
    """CIFAR-native ResNet must accept 32x32 input directly (no upsampling)."""
    from earlyon.models import cifar_resnet_ee

    model = cifar_resnet_ee(num_classes=10, depth=20)
    model.eval()

    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        result = model(x, mode="inference")
    assert result.prediction.shape == (1, 10)
    # three stages → three exit points
    assert len(model.config.exit_points) == 3


def test_cifar_resnet_ee_training_returns_all_exit_logits():
    from earlyon.models import cifar_resnet_ee

    model = cifar_resnet_ee(num_classes=10, depth=20)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        outs = model(x, mode="training")
    # 3 exits + final
    assert len(outs) == 4
    for o in outs:
        assert o.shape == (2, 10)


def test_cifar_resnet_ee_identity_invariant():
    from earlyon.models import cifar_resnet_ee

    model = cifar_resnet_ee(num_classes=10, depth=20)
    model.config.confidence_thresholds = [1.0, 1.0, 1.0]
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        result = model(x, mode="inference")
        direct = model.backbone(x)
    assert result.exit_taken == -1
    assert torch.allclose(result.prediction, direct, atol=1e-5)


@pytest.mark.parametrize("depth", [20, 32, 56])
def test_cifar_resnet_ee_supports_he2015_depths(depth):
    """He et al. 2015 CIFAR ResNets parameterize depth as 6n+2."""
    from earlyon.models import cifar_resnet_ee

    model = cifar_resnet_ee(num_classes=10, depth=depth)
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        result = model(x, mode="inference")
    assert result.prediction.shape == (1, 10)


def test_cifar_resnet_ee_rejects_invalid_depth():
    """Depths that are not 6n+2 must raise."""
    from earlyon.models import cifar_resnet_ee

    with pytest.raises(ValueError, match="6n"):
        cifar_resnet_ee(num_classes=10, depth=21)
