"""Wrap any ``nn.Module`` with early exits at named submodules.

Feature widths are auto-inferred via one dry-run forward pass with temporary
hooks, so callers don't hand-specify channel counts. This is the
``earlyon.models`` entry point for arbitrary user-provided backbones.

Custom-wrapped models are NOT round-trippable through ``save_wrapper`` /
``load_wrapper`` (``build_model`` cannot reconstruct an arbitrary backbone from a
string); ``build_model`` raises a clear error for ``backbone="custom"``. If you
save/load the ``state_dict`` yourself, rebuild the heads with the same
``pool_tokens`` you trained with — it is not stored in the weights, and a
mismatch loads silently while changing how 3D token features are pooled.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn

from earlyon.core.exit_head import EarlyExitHead
from earlyon.core.types import EarlyExitConfig, ExitPoint
from earlyon.core.wrappers import EarlyExitWrapper
from earlyon.models._common import identity


def _infer_feature_widths(
    backbone: nn.Module,
    layer_names: Sequence[str],
    input_shape: tuple[int, int, int, int],
) -> dict[str, int]:
    """Run one no-grad dry-run forward with temporary hooks to record each exit
    layer's feature width (``C`` for 4D output, ``D`` for 3D/2D).

    Raises ``ValueError`` if a layer produces a non-Tensor or unsupported rank,
    ``RuntimeError`` if a layer is never visited during the forward pass, and
    ``AttributeError`` (from ``get_submodule``) for an unresolvable layer name.
    Restores the backbone's prior train/eval mode on exit.
    """
    widths: dict[str, int] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_probe(name: str) -> Callable[[nn.Module, object, object], None]:
        def hook(module: nn.Module, inputs: object, output: object) -> None:
            if not isinstance(output, torch.Tensor):
                raise ValueError(
                    f"exit layer {name!r} produced {type(output).__name__}, not a "
                    "Tensor — custom_ee cannot auto-infer its feature width"
                )
            if output.dim() == 4:
                widths[name] = output.shape[1]  # (B, C, H, W) -> C
            elif output.dim() in (2, 3):
                widths[name] = output.shape[-1]  # (B, D) / (B, N, D) -> D
            else:
                raise ValueError(
                    f"exit layer {name!r} output has unsupported rank {output.dim()} "
                    f"(shape {tuple(output.shape)}); supported: 2D, 3D, or 4D"
                )

        return hook

    for name in layer_names:
        module = backbone.get_submodule(name)  # AttributeError on bad name
        handles.append(module.register_forward_hook(_make_probe(name)))

    was_training = backbone.training
    backbone.eval()
    try:
        with torch.no_grad():
            backbone(torch.zeros(input_shape))
    finally:
        for handle in handles:
            handle.remove()
        backbone.train(was_training)

    missing = [name for name in layer_names if name not in widths]
    if missing:
        raise RuntimeError(
            f"dry-run forward never visited exit layers {missing}; verify they are "
            f"reachable for input_shape={input_shape}"
        )
    return widths


def custom_ee(
    backbone: nn.Module,
    exit_layers: Sequence[str],
    num_classes: int,
    *,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    hidden_dim: int = 128,
    dropout: float = 0.2,
    pool_tokens: str = "mean",
    confidence_thresholds: Sequence[float] | None = None,
    routing_policy: str = "confidence",
) -> EarlyExitWrapper:
    """Wrap any ``nn.Module`` with early exits at the named submodules.

    Feature widths are auto-inferred from a single dry-run forward, so you only
    name the layers. ``exit_layers`` order is the exit order — the names must be
    in forward-execution order (``exit_layers[0]`` runs first), since routing
    short-circuits at the first confident exit.

    The caller is responsible for making ``backbone(x)`` return
    ``(B, num_classes)`` logits (e.g. replace the backbone's final head before
    wrapping); ``final_classifier`` is identity.

    Parameters
    ----------
    backbone:
        Any ``nn.Module`` callable on ``input_shape``-shaped input.
    exit_layers:
        Dotted submodule names resolvable via ``backbone.get_submodule``, in
        forward order.
    num_classes:
        Output classes for every exit head.
    input_shape:
        Shape for the dry-run forward and FLOPs accounting.
    pool_tokens:
        For 3D token features, ``"mean"`` (default, architecture-agnostic) or
        ``"cls"`` (token 0). Ignored for 4D conv features.
    confidence_thresholds:
        Per-exit thresholds; defaults to ``[0.8] * len(exit_layers)``.
    routing_policy:
        ``"confidence"`` or ``"entropy"``.
    """
    if len(exit_layers) == 0:
        raise ValueError("exit_layers must be non-empty")

    widths = _infer_feature_widths(backbone, exit_layers, input_shape)
    exit_points = [ExitPoint(f"e{i}", name, widths[name]) for i, name in enumerate(exit_layers)]
    heads: dict[str, nn.Module] = {
        ep.name: EarlyExitHead(
            ep.in_channels,
            num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pool_tokens=pool_tokens,
        )
        for ep in exit_points
    }
    cfg = EarlyExitConfig(
        backbone="custom",
        num_classes=num_classes,
        exit_points=exit_points,
        confidence_thresholds=list(confidence_thresholds) if confidence_thresholds else [],
        routing_policy=routing_policy,
    )
    return EarlyExitWrapper(backbone, heads, identity, cfg, input_shape=input_shape)
