"""Core dataclasses used across earlyon."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class ExitPoint:
    """Describes one early-exit attachment point on a backbone.

    layer_name is a dotted path resolvable via ``model.get_submodule(name)``
    (e.g. ``"layer2"`` for ResNet, ``"features.10"`` for MobileNetV2).
    """

    name: str
    layer_name: str
    in_channels: int


@dataclass
class EarlyExitConfig:
    """Per-model configuration. ``loss_weights`` includes the final classifier
    at index -1, so its length is ``len(exit_points) + 1``.

    ``routing_policy`` selects how the wrapper decides whether to exit at an
    intermediate head:

    * ``"confidence"`` — exit when ``softmax(logits).max() >= threshold``.
      Uses ``confidence_thresholds``.
    * ``"entropy"`` — exit when ``H(softmax(logits)) <= threshold`` (low
      entropy = high confidence). Uses ``entropy_thresholds``.
    """

    backbone: str
    num_classes: int
    exit_points: list[ExitPoint]
    loss_weights: list[float] = field(default_factory=list)
    confidence_thresholds: list[float] = field(default_factory=list)
    entropy_thresholds: list[float] = field(default_factory=list)
    routing_policy: str = "confidence"
    temperature: float = 1.0

    def __post_init__(self) -> None:
        n_exits = len(self.exit_points)
        if not self.loss_weights:
            # equal weight across exits + final
            w = 1.0 / (n_exits + 1)
            self.loss_weights = [w] * (n_exits + 1)
        if not self.confidence_thresholds:
            self.confidence_thresholds = [0.8] * n_exits
        if not self.entropy_thresholds:
            # half the max possible entropy for `num_classes`-way uniform
            default = 0.5 * math.log(max(self.num_classes, 2))
            self.entropy_thresholds = [default] * n_exits
        if len(self.loss_weights) != n_exits + 1:
            raise ValueError(
                f"loss_weights must have length {n_exits + 1} (got {len(self.loss_weights)})"
            )
        if len(self.confidence_thresholds) != n_exits:
            raise ValueError(
                f"confidence_thresholds must have length {n_exits} "
                f"(got {len(self.confidence_thresholds)})"
            )
        if len(self.entropy_thresholds) != n_exits:
            raise ValueError(
                f"entropy_thresholds must have length {n_exits} "
                f"(got {len(self.entropy_thresholds)})"
            )
        if self.routing_policy not in {"confidence", "entropy"}:
            raise ValueError(
                f"routing_policy={self.routing_policy!r} not supported "
                "(allowed: 'confidence', 'entropy')"
            )


@dataclass
class InferenceResult:
    """Result of a single-sample inference pass.

    ``exit_taken`` is the 0-indexed exit number; -1 means the final classifier
    was used. ``computation_used`` is the FLOPs fraction (not layer count) of
    the network actually executed.
    """

    prediction: torch.Tensor
    exit_taken: int
    confidence: float
    computation_used: float


@dataclass
class BatchedInferenceResult:
    """Result of a batched inference pass with per-batch routing.

    All samples in the batch exit together at ``exit_taken``. The
    ``per_sample_confidence`` tensor records each sample's confidence at the
    exit point.
    """

    predictions: torch.Tensor
    exit_taken: int
    per_sample_confidence: torch.Tensor
    computation_used: float
