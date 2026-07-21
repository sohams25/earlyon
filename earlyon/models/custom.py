"""Wrap any ``nn.Module`` with early exits at named submodules.

Feature widths are auto-inferred via one dry-run forward pass with temporary
hooks, so callers don't hand-specify channel counts. This is the
``earlyon.models`` entry point for arbitrary user-provided backbones.

The dry run also *validates* the wrapping: every requested exit layer must
execute exactly once, in the order the caller listed them — a reused layer
or an out-of-order exit list would silently mis-route at inference, so both
fail loudly here instead.

Runtime contract: the wrapper's inference/training paths call
``backbone(x)`` with a single tensor. ``example_args``/``example_kwargs``
configure the *inspection* forward only — use them when the backbone needs
extra arguments that have defaults at runtime, or a non-default example
tensor. The example (or ``input_shape``) tensor is created on the backbone's
own device unless ``device`` says otherwise.

Custom-wrapped models are reloadable only via a user-provided factory:
``load_wrapper(path, factory=lambda: custom_ee(...))`` with the same
structure (including ``pool_tokens`` and any ``feature_extractors`` — neither
is stored in the weights).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn

from earlyon.core.exit_head import EarlyExitHead
from earlyon.core.types import EarlyExitConfig, ExitPoint
from earlyon.core.wrappers import EarlyExitWrapper
from earlyon.models._common import identity

FeatureExtractor = Callable[[Any], torch.Tensor]


def _infer_backbone_device(backbone: nn.Module) -> torch.device:
    for p in backbone.parameters():
        return p.device
    for b in backbone.buffers():
        return b.device
    return torch.device("cpu")


def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    return obj


class _DryRunReport:
    def __init__(self) -> None:
        self.widths: dict[str, int] = {}
        self.execution_order: list[str] = []
        self.call_counts: dict[str, int] = {}


def _dry_run(
    backbone: nn.Module,
    layer_names: Sequence[str],
    example_args: tuple[Any, ...],
    example_kwargs: dict[str, Any],
    extractors: Mapping[str, FeatureExtractor],
) -> _DryRunReport:
    """One no-grad forward with temporary hooks: infer each exit layer's
    feature width, record execution order and per-layer call counts.

    Raises ``ValueError`` for an unresolvable layer name, a non-Tensor output
    without an extractor, or an unsupported rank; ``RuntimeError`` for layers
    never visited, visited more than once, or visited out of order. Restores
    the backbone's prior train/eval mode on exit.
    """
    report = _DryRunReport()
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_probe(name: str) -> Callable[[nn.Module, object, object], None]:
        def hook(module: nn.Module, inputs: object, output: object) -> None:
            report.call_counts[name] = report.call_counts.get(name, 0) + 1
            report.execution_order.append(name)
            extractor = extractors.get(name)
            features = extractor(output) if extractor is not None else output
            if not isinstance(features, torch.Tensor):
                hint = (
                    "its feature_extractors entry must return a Tensor"
                    if extractor is not None
                    else "pass feature_extractors={name: fn} to convert it to a Tensor"
                )
                raise ValueError(
                    f"exit layer {name!r} produced {type(features).__name__}, not a "
                    f"Tensor — {hint}"
                )
            if features.dim() == 4:
                report.widths[name] = features.shape[1]  # (B, C, H, W) -> C
            elif features.dim() in (2, 3):
                report.widths[name] = features.shape[-1]  # (B, D) / (B, N, D) -> D
            else:
                raise ValueError(
                    f"exit layer {name!r} features have unsupported rank {features.dim()} "
                    f"(shape {tuple(features.shape)}); supported: 2D, 3D, or 4D. "
                    "Use feature_extractors to reshape."
                )

        return hook

    for name in layer_names:
        try:
            module = backbone.get_submodule(name)
        except AttributeError:
            available = [n for n, _ in backbone.named_modules() if n]
            shown = ", ".join(available[:20]) + (", ..." if len(available) > 20 else "")
            raise ValueError(
                f"exit layer {name!r} does not exist on the backbone; " f"available layers: {shown}"
            ) from None
        handles.append(module.register_forward_hook(_make_probe(name)))

    was_training = backbone.training
    backbone.eval()
    try:
        with torch.no_grad():
            backbone(*example_args, **example_kwargs)
    finally:
        for handle in handles:
            handle.remove()
        backbone.train(was_training)

    missing = [name for name in layer_names if name not in report.widths]
    if missing:
        raise RuntimeError(
            f"dry-run forward never visited exit layers {missing}; verify they are "
            "reachable for the example input"
        )
    reused = [name for name, c in report.call_counts.items() if c > 1]
    if reused:
        raise RuntimeError(
            f"exit layer(s) {reused} execute more than once per forward pass "
            "(module reuse); early-exit routing at a reused layer is ambiguous "
            "and unsupported"
        )
    if report.execution_order != list(layer_names):
        raise RuntimeError(
            f"exit_layers must be listed in forward-execution order; the dry run "
            f"observed {report.execution_order} but you passed {list(layer_names)}. "
            "Routing short-circuits at the first confident exit, so order matters."
        )
    return report


def custom_ee(
    backbone: nn.Module,
    exit_layers: Sequence[str],
    num_classes: int,
    *,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    example_args: tuple[Any, ...] | None = None,
    example_kwargs: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
    feature_extractors: Mapping[str, FeatureExtractor] | None = None,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    pool_tokens: str = "mean",
    confidence_thresholds: Sequence[float] | None = None,
    routing_policy: str = "confidence",
) -> EarlyExitWrapper:
    """Wrap any ``nn.Module`` with early exits at the named submodules.

    Feature widths are auto-inferred from a single dry-run forward, which also
    validates that each exit layer runs exactly once and in the listed order.

    The caller is responsible for making ``backbone(x)`` return
    ``(B, num_classes)`` logits at runtime (e.g. replace the backbone's final
    head before wrapping); ``final_classifier`` is identity.

    Parameters
    ----------
    backbone:
        Any ``nn.Module``. The dry run calls it on the example input; at
        runtime the wrapper calls ``backbone(x)`` with a single tensor.
    exit_layers:
        Dotted submodule names resolvable via ``backbone.get_submodule``, in
        forward-execution order (validated by the dry run).
    num_classes:
        Output classes for every exit head.
    input_shape:
        Shape for the default dry-run tensor and the FLOPs estimate. Ignored
        for the dry run when ``example_args`` is given.
    example_args / example_kwargs:
        Explicit example inputs for the dry run — for backbones needing more
        than ``zeros(input_shape)`` (extra positional/keyword arguments with
        runtime defaults, integer token ids, ...). Tensors are moved to the
        inspection device.
    device:
        Where to run the dry run. Default: the backbone's own parameter
        device — a CUDA model is no longer probed with a CPU tensor.
    feature_extractors:
        Optional per-*layer-name* callables converting a layer's raw output
        (tuple, dict, unusual rank) into the 2D/3D/4D Tensor its exit head
        consumes. Also applied at inference; not serialized — rebuild them in
        your ``load_wrapper`` factory.
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

    run_device = torch.device(device) if device is not None else _infer_backbone_device(backbone)
    if example_args is None and example_kwargs:
        raise ValueError("example_kwargs requires example_args (may be an empty tuple)")
    if example_args is None:
        example_args = (torch.zeros(input_shape, device=run_device),)
        example_kwargs = {}
    else:
        example_args = tuple(_to_device(a, run_device) for a in example_args)
        example_kwargs = {k: _to_device(v, run_device) for k, v in (example_kwargs or {}).items()}

    extractors = dict(feature_extractors or {})
    unknown = [k for k in extractors if k not in set(exit_layers)]
    if unknown:
        raise ValueError(f"feature_extractors keyed by unknown exit layer(s): {unknown}")

    report = _dry_run(backbone, exit_layers, example_args, example_kwargs, extractors)

    exit_points = [
        ExitPoint(f"e{i}", name, report.widths[name]) for i, name in enumerate(exit_layers)
    ]
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
    # feature extractors are keyed by layer name for the user; the wrapper
    # keys adapters by exit name.
    adapters = {
        ep.name: extractors[ep.layer_name] for ep in exit_points if ep.layer_name in extractors
    }
    wrapper = EarlyExitWrapper(
        backbone, heads, identity, cfg, input_shape=input_shape, feature_adapters=adapters
    )
    # the fresh exit heads must live where the backbone lives, or a wrapped
    # CUDA model fails at the first routed inference with a device mismatch
    return wrapper.to(run_device)
