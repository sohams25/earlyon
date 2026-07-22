"""Core dataclasses used across earlyon."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

# A (images, targets) minibatch — the element type of every earlyon DataLoader.
Batch = tuple[torch.Tensor, torch.Tensor]

# Reserved key for the final classifier head in per-head mappings
# (``EarlyExitConfig.temperatures``). Exit points may not use this name.
FINAL_HEAD = "final"

VALID_ROUTING_POLICIES = ("confidence", "entropy")


def exit_label(idx: int) -> str:
    """Canonical name for an exit index: ``"final"`` for -1, else ``"exit_{idx}"``."""
    return FINAL_HEAD if idx == -1 else f"exit_{idx}"


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

    ``enabled_exits`` is the explicit per-exit on/off switch. A disabled exit
    never fires, regardless of thresholds — including the numerical edge cases
    (softmax confidence saturated at exactly 1.0, entropy at exactly 0.0) that
    made the old "threshold sentinel means disabled" convention unsound.

    ``temperatures`` maps each head — every ``ExitPoint.name`` plus the
    reserved key ``"final"`` — to its softmax temperature (Guo et al. 2017).
    Per-head temperatures matter because each exit head is a different
    classifier with its own miscalibration. The legacy scalar ``temperature``
    field is a constructor convenience only: it is broadcast into
    ``temperatures`` at construction time and ignored afterwards; mutate
    ``temperatures`` (not ``temperature``) to change calibration.
    """

    backbone: str
    num_classes: int
    exit_points: list[ExitPoint]
    loss_weights: list[float] = field(default_factory=list)
    confidence_thresholds: list[float] = field(default_factory=list)
    entropy_thresholds: list[float] = field(default_factory=list)
    enabled_exits: list[bool] = field(default_factory=list)
    routing_policy: str = "confidence"
    temperature: float = 1.0  # legacy scalar; broadcast into `temperatures`
    temperatures: dict[str, float] = field(default_factory=dict)

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
        if not self.enabled_exits:
            self.enabled_exits = [True] * n_exits
        if not self.temperatures:
            # migrate the legacy scalar deterministically: same T on every head
            self.temperatures = {ep.name: self.temperature for ep in self.exit_points}
            self.temperatures[FINAL_HEAD] = self.temperature
        elif self.temperature != 1.0:
            raise ValueError(
                "pass either the legacy scalar `temperature` or the per-head "
                "`temperatures` mapping, not both"
            )
        self.validate()

    # ---------------- helpers ----------------

    @property
    def head_names(self) -> list[str]:
        """Every head that produces logits: exit names in order, then ``"final"``."""
        return [ep.name for ep in self.exit_points] + [FINAL_HEAD]

    def temperature_for(self, head_name: str) -> float:
        """The softmax temperature for ``head_name`` (an exit name or ``"final"``)."""
        return self.temperatures[head_name]

    def validate(self) -> None:
        """Check every configuration invariant; raise ``ValueError`` with an
        actionable message on the first violation.

        Called automatically at construction. Because this dataclass is mutable
        (calibration writes fitted values back), ``EarlyExitWrapper`` and the
        checkpoint loader re-invoke it at their own trust boundaries.
        """
        if self.num_classes < 2:
            # with a single class softmax confidence is identically 1.0, so
            # every exit fires at any threshold <= 1.0 — routing is meaningless
            raise ValueError(f"num_classes must be >= 2 (got {self.num_classes})")

        n_exits = len(self.exit_points)
        names = [ep.name for ep in self.exit_points]
        layers = [ep.layer_name for ep in self.exit_points]
        if any(not n for n in names) or any(not layer for layer in layers):
            raise ValueError("exit point names and layer names must be non-empty")
        if len(set(names)) != n_exits:
            raise ValueError(f"exit point names must be unique (got {names})")
        if len(set(layers)) != n_exits:
            raise ValueError(f"exit point layer names must be unique (got {layers})")
        if FINAL_HEAD in names:
            raise ValueError(f"exit point name {FINAL_HEAD!r} is reserved for the final head")

        if self.routing_policy not in VALID_ROUTING_POLICIES:
            raise ValueError(
                f"routing_policy={self.routing_policy!r} not supported "
                f"(allowed: {', '.join(repr(p) for p in VALID_ROUTING_POLICIES)})"
            )

        if len(self.loss_weights) != n_exits + 1:
            raise ValueError(
                f"loss_weights must have length {n_exits + 1} (got {len(self.loss_weights)})"
            )
        if any(not math.isfinite(w) or w < 0 for w in self.loss_weights):
            raise ValueError(f"loss_weights must be finite and >= 0 (got {self.loss_weights})")
        if sum(self.loss_weights) <= 0:
            raise ValueError("loss_weights must not all be zero")

        if len(self.confidence_thresholds) != n_exits:
            raise ValueError(
                f"confidence_thresholds must have length {n_exits} "
                f"(got {len(self.confidence_thresholds)})"
            )
        if any(not math.isfinite(t) or not 0.0 <= t <= 1.0 for t in self.confidence_thresholds):
            raise ValueError(
                "confidence_thresholds must be finite and in [0, 1] "
                f"(got {self.confidence_thresholds}); to disable an exit set "
                "enabled_exits[i] = False instead of using a sentinel threshold"
            )
        if len(self.entropy_thresholds) != n_exits:
            raise ValueError(
                f"entropy_thresholds must have length {n_exits} "
                f"(got {len(self.entropy_thresholds)})"
            )
        if any(not math.isfinite(t) or t < 0.0 for t in self.entropy_thresholds):
            raise ValueError(
                f"entropy_thresholds must be finite and >= 0 (got {self.entropy_thresholds}); "
                "to disable an exit set enabled_exits[i] = False instead of a sentinel"
            )

        if len(self.enabled_exits) != n_exits:
            raise ValueError(
                f"enabled_exits must have length {n_exits} (got {len(self.enabled_exits)})"
            )
        if any(not isinstance(e, bool) for e in self.enabled_exits):
            raise ValueError(f"enabled_exits must be booleans (got {self.enabled_exits})")

        expected_keys = set(names) | {FINAL_HEAD}
        if set(self.temperatures) != expected_keys:
            raise ValueError(
                f"temperatures must have exactly one entry per head "
                f"({sorted(expected_keys)}); got keys {sorted(self.temperatures)}"
            )
        for head, t in self.temperatures.items():
            if not math.isfinite(t) or t <= 0.0:
                raise ValueError(
                    f"temperatures[{head!r}] must be finite and > 0 (got {t}). "
                    "A fitted temperature this bad means calibration diverged; "
                    "refit or fall back to 1.0 explicitly."
                )


def migrate_scalar_temperature(config: EarlyExitConfig, temperature: float) -> None:
    """Deterministically broadcast a legacy scalar temperature onto every head.

    Used by the v1-checkpoint migration path; emits no warning itself (the
    caller decides how loudly to announce the migration).
    """
    config.temperatures = {ep.name: temperature for ep in config.exit_points}
    config.temperatures[FINAL_HEAD] = temperature
    config.validate()


@dataclass
class InferenceResult:
    """Result of a single-sample inference pass.

    ``exit_taken`` is the 0-indexed exit number; -1 means the final classifier
    was used.

    ``estimated_backbone_flops_fraction`` is an *estimate* of the fraction of
    the backbone's FLOPs executed, derived from a one-time static analysis of
    the backbone (see ``earlyon.core.flops``). It is not a measurement of this
    inference, and it excludes the exit heads' own (small) cost and all
    routing overhead. ``computation_used`` remains as a read alias for
    backward compatibility.

    ``prediction`` is produced under ``torch.inference_mode()`` (the deployment
    path is grad-free by contract), so it is an *inference tensor*: ready for
    ``argmax``/``item``/serialization, but not usable in a tracked autograd
    computation. For gradient-requiring work use ``model(x, mode="training")``,
    or ``prediction.clone()`` *after* this call returns to detach it.
    """

    prediction: torch.Tensor
    exit_taken: int
    confidence: float
    estimated_backbone_flops_fraction: float

    @property
    def computation_used(self) -> float:
        """Deprecated alias for ``estimated_backbone_flops_fraction``."""
        return self.estimated_backbone_flops_fraction


@dataclass
class BatchedInferenceResult:
    """Result of a batched inference pass with per-batch routing.

    All samples in the batch exit together at ``exit_taken``. The
    ``per_sample_confidence`` tensor records each sample's confidence at the
    exit point. ``estimated_backbone_flops_fraction`` follows the same
    estimator contract as :class:`InferenceResult`.
    """

    predictions: torch.Tensor
    exit_taken: int
    per_sample_confidence: torch.Tensor
    estimated_backbone_flops_fraction: float

    @property
    def computation_used(self) -> float:
        """Deprecated alias for ``estimated_backbone_flops_fraction``."""
        return self.estimated_backbone_flops_fraction
