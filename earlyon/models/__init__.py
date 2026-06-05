"""Pre-configured early-exit model factories."""

from earlyon.models.cifar_resnet_ee import cifar_resnet_ee
from earlyon.models.custom import custom_ee
from earlyon.models.efficientnet_ee import efficientnet_b0_ee
from earlyon.models.mobilenetv2_ee import mobilenetv2_ee
from earlyon.models.resnet_ee import resnet18_ee, resnet50_ee
from earlyon.models.vit_ee import vit_b_16_ee

__all__ = [
    "resnet18_ee",
    "resnet50_ee",
    "mobilenetv2_ee",
    "efficientnet_b0_ee",
    "cifar_resnet_ee",
    "vit_b_16_ee",
    "custom_ee",
]
