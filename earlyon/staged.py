"""Staged deployment: split an early-exit model into per-stage modules.

The eager :class:`EarlyExitWrapper` short-circuits with hooks and exceptions,
which no graph exporter can express. The staged contract is the deployable
alternative: the network is cut *at the exit points* into an ordered list of
stages, where stage ``k``:

* consumes the previous stage's **continuation features** (stage 0 consumes
  the input image),
* produces the next continuation features, and
* produces **exit logits** (from exit head ``k``) for every stage except the
  last, which produces the final classifier's logits.

The deployment runtime (this module's :class:`StagedModel` in PyTorch, or
your own runtime over per-stage ONNX/TensorRT engines) runs stages in order
and applies earlyon's routing rule between them — per-head temperature,
enabled flag, confidence/entropy threshold — stopping at the first firing
exit. Unlike the single static multi-output ONNX graph
(:func:`earlyon.onnx.export_to_onnx`), later stages are genuinely *not
executed* when an early exit fires.

Scope (deliberately narrow): the reference splitter supports backbones that
are literally ``nn.Sequential`` with exit layers at top-level positions —
arbitrary graph partitioning is out of scope and fails loudly.
:func:`staged_model` verifies its own equivalence against the eager wrapper
on a probe input before returning. See ``docs/STAGED_DEPLOYMENT.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn

from earlyon.core.types import EarlyExitConfig, InferenceResult
from earlyon.core.wrappers import EarlyExitWrapper, _safe_temperature


@dataclass(frozen=True)
class StageSpec:
    """Describes one stage of a staged deployment.

    ``exit_name`` is the exit whose logits this stage emits (``"final"`` for
    the last stage). ``modules`` lists the backbone submodule names executed
    by the stage, in order.
    """

    index: int
    exit_name: str
    modules: tuple[str, ...]


class Stage(nn.Module):
    """One deployable unit: a trunk plus the head evaluated after it.

    ``forward`` returns ``(continuation_features, logits)``. For the last
    stage the continuation features equal the logits input path's features
    and ``logits`` are the final classifier's output.
    """

    def __init__(self, trunk: nn.Sequential, head: nn.Module) -> None:
        super().__init__()
        self.trunk = trunk
        self.head = head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        return features, self.head(features)


class _Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class StagedModel(nn.Module):
    """Reference staged runtime: runs stages in order, applies earlyon routing
    between them, and stops at the first firing enabled exit.

    Produces exactly the same :class:`InferenceResult` as the eager wrapper
    for the same config (equivalence is asserted at build time by
    :func:`staged_model` and pinned by tests).
    """

    def __init__(
        self,
        stages: list[Stage],
        specs: list[StageSpec],
        config: EarlyExitConfig,
        flops_at: dict[str, float],
    ) -> None:
        super().__init__()
        if len(stages) != len(config.exit_points) + 1:
            raise ValueError(
                f"need one stage per exit plus a final stage: got {len(stages)} stages "
                f"for {len(config.exit_points)} exits"
            )
        config.validate()
        self.stages = nn.ModuleList(stages)
        self.specs = specs
        self.config = config
        self._flops_at = dict(flops_at)

    @torch.inference_mode()
    def infer(self, x: torch.Tensor) -> InferenceResult:
        if x.size(0) != 1:
            raise ValueError(f"staged inference is batch-1 (got batch {x.size(0)})")
        cfg = self.config
        n_exits = len(cfg.exit_points)
        features = x
        for idx in range(n_exits):
            ep = cfg.exit_points[idx]
            if not cfg.enabled_exits[idx]:
                # skip the head entirely; still run the trunk to continue
                stage = cast(Stage, self.stages[idx])
                features = stage.trunk(features)
                continue
            features, logits = self.stages[idx](features)
            temp = _safe_temperature(cfg.temperatures[ep.name])
            probs = torch.softmax(logits / temp, dim=-1)
            confidence = probs.max().item()
            if cfg.routing_policy == "confidence":
                fires = confidence >= cfg.confidence_thresholds[idx]
            else:
                entropy = (-(probs * probs.clamp_min(1e-12).log()).sum()).item()
                fires = entropy <= cfg.entropy_thresholds[idx]
            if fires:
                return InferenceResult(
                    prediction=logits,
                    exit_taken=idx,
                    confidence=confidence,
                    estimated_backbone_flops_fraction=self._flops_at[ep.layer_name],
                )
        _, final_logits = self.stages[-1](features)
        temp = _safe_temperature(cfg.temperatures["final"])
        probs = torch.softmax(final_logits / temp, dim=-1)
        return InferenceResult(
            prediction=final_logits,
            exit_taken=-1,
            confidence=probs.max().item(),
            estimated_backbone_flops_fraction=1.0,
        )


def _sequential_children(backbone: nn.Module) -> list[tuple[str, nn.Module]]:
    if not isinstance(backbone, nn.Sequential):
        raise ValueError(
            "staged_model supports backbones that are nn.Sequential (exit layers "
            f"at top level); got {type(backbone).__name__}. Arbitrary graph "
            "partitioning is out of scope — see docs/STAGED_DEPLOYMENT.md for the "
            "protocol if you want to implement stages for a custom architecture."
        )
    return list(backbone.named_children())


def staged_model(wrapper: EarlyExitWrapper) -> StagedModel:
    """Split an eager wrapper into a :class:`StagedModel`.

    Requirements (all verified):

    * the wrapper's backbone is an ``nn.Sequential`` whose forward is exactly
      its children in order, returning ``(B, num_classes)`` logits (i.e. the
      wrapper's ``final_classifier`` is identity — true for ``custom_ee``);
    * every exit layer is a *top-level* child of that Sequential.

    Before returning, staged inference is checked against the eager wrapper
    on a probe input; a mismatch raises rather than shipping a wrong split.
    """
    children = _sequential_children(wrapper.backbone)
    child_names = [name for name, _ in children]
    positions: list[int] = []
    for ep in wrapper.config.exit_points:
        if ep.layer_name not in child_names:
            raise ValueError(
                f"exit layer {ep.layer_name!r} is not a top-level child of the "
                f"Sequential backbone (children: {child_names}); staged splitting "
                "requires top-level exit layers"
            )
        positions.append(child_names.index(ep.layer_name))
    if positions != sorted(positions):
        raise ValueError(f"exit layers out of order in the Sequential: {positions}")

    stages: list[Stage] = []
    specs: list[StageSpec] = []
    start = 0
    for idx, ep in enumerate(wrapper.config.exit_points):
        end = positions[idx] + 1
        trunk = nn.Sequential(*(mod for _, mod in children[start:end]))
        stages.append(Stage(trunk, wrapper.exit_heads[ep.name]))
        specs.append(StageSpec(idx, ep.name, tuple(child_names[start:end])))
        start = end
    final_trunk = nn.Sequential(*(mod for _, mod in children[start:]))
    stages.append(Stage(final_trunk, _Identity()))
    specs.append(StageSpec(len(positions), "final", tuple(child_names[start:])))

    model = StagedModel(stages, specs, wrapper.config, wrapper._flops_at)

    # equivalence probe: staged and eager must agree on a deterministic input.
    was_training = wrapper.training
    wrapper.eval()
    model.eval()
    try:
        probe = torch.zeros(
            wrapper._input_shape,
            device=next(wrapper.backbone.parameters(), torch.empty(0)).device,
        )
        eager = wrapper(probe, mode="inference")
        staged = model.infer(probe)
        if eager.exit_taken != staged.exit_taken or not torch.allclose(
            eager.prediction, staged.prediction, atol=1e-5
        ):
            raise RuntimeError(
                "staged split does not reproduce the eager wrapper on a probe "
                "input — the backbone's forward is not exactly its Sequential "
                "children in order; refusing to build a wrong split"
            )
    finally:
        wrapper.train(was_training)
    return model
