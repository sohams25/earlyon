"""earlyon — early exit toolkit for PyTorch CV models."""

from earlyon.core.types import BatchedInferenceResult, EarlyExitConfig, ExitPoint, InferenceResult
from earlyon.core.exit_head import EarlyExitHead
from earlyon.core.wrappers import EarlyExitWrapper

__version__ = "0.1.0"

__all__ = [
    "EarlyExitWrapper",
    "EarlyExitHead",
    "EarlyExitConfig",
    "ExitPoint",
    "InferenceResult",
    "BatchedInferenceResult",
    "__version__",
]
