"""Pre-configured early-exit model factories."""

from earlyon.models.efficientnet_ee import efficientnet_b0_ee
from earlyon.models.mobilenetv2_ee import mobilenetv2_ee
from earlyon.models.resnet_ee import resnet18_ee, resnet50_ee

__all__ = ["resnet18_ee", "resnet50_ee", "mobilenetv2_ee", "efficientnet_b0_ee"]
