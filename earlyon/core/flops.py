"""Static FLOPs *estimation* helpers.

What this module produces is an estimate, not a measurement. The estimator
(fvcore's ``FlopCountAnalysis``) statically attributes FLOPs per module; the
cumulative fraction at an exit layer is derived from module *registration
order* filtered to leaves, which matches execution order for straight-line
(sequential-style) backbones — the torchvision CNNs and ViT earlyon ships.

Known limitations (all surfaced, none silent):

* **Exit-head FLOPs are excluded.** Only the backbone is analysed; the small
  exit heads (~10-100k params) and all routing overhead are not counted.
* **Reused modules break the ordering assumption.** If any leaf module runs
  more than once in a forward pass (weight sharing, recursive blocks), a
  cumulative "up to layer L" sum over registration order is not meaningful.
  The probe detects this and returns a *low-confidence uniform estimate*
  (``method="uniform-fallback"``, ``reliable=False``) with a warning, instead
  of a precise-looking but wrong fraction.
* fvcore does not count some ops (e.g. scaled-dot-product attention matmuls),
  so ViT fractions are a slight under-count; they remain monotone.

The analysis costs one or two forward passes of the backbone, so callers
(``EarlyExitWrapper``) run it lazily on first use, not at construction.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn

METHOD_FVCORE = "fvcore-leaf-walk"
METHOD_UNIFORM = "uniform-fallback"


@dataclass(frozen=True)
class FlopsEstimate:
    """Cumulative backbone-FLOPs fractions per exit layer, with provenance.

    ``fractions[layer]`` is (estimated FLOPs of everything executed up to and
    including ``layer``) / (total estimated backbone FLOPs), in [0, 1].
    ``reliable`` is False when the estimator had to fall back to uniform
    spacing (fvcore missing, zero total, or a reused/multi-call module that
    invalidates the ordering assumption); treat those fractions as layout
    placeholders, not compute estimates.
    """

    fractions: dict[str, float]
    method: str
    reliable: bool
    notes: tuple[str, ...] = field(default_factory=tuple)
    excludes_exit_heads: bool = True


def _uniform(layer_names: Sequence[str], note: str) -> FlopsEstimate:
    n = len(layer_names)
    return FlopsEstimate(
        fractions={name: (i + 1) / (n + 1) for i, name in enumerate(layer_names)},
        method=METHOD_UNIFORM,
        reliable=False,
        notes=(note,),
    )


def _count_leaf_calls(
    backbone: nn.Module, input_shape: tuple[int, int, int, int], device: str
) -> dict[str, int]:
    """One dummy forward counting how many times each leaf module executes."""
    counts: dict[str, int] = {}
    handles = []
    for name, mod in backbone.named_modules():
        if name and len(list(mod.children())) == 0:
            counts[name] = 0

            def _hook(_m: nn.Module, _i: object, _o: object, _name: str = name) -> None:
                counts[_name] += 1

            handles.append(mod.register_forward_hook(_hook))
    try:
        with torch.no_grad():
            backbone(torch.zeros(input_shape, device=device))
    finally:
        for h in handles:
            h.remove()
    return counts


def estimate_layer_flops(
    backbone: nn.Module,
    layer_names: Sequence[str],
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = "cpu",
) -> FlopsEstimate:
    """Estimate the cumulative FLOPs fraction at each named layer.

    ``device`` is where the probe input is created — it must match the
    backbone's parameters (the wrapper passes its backbone's current device,
    since the estimate is computed lazily and the model may have moved).

    See the module docstring for the estimator's assumptions and limitations.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return _uniform(layer_names, "fvcore not installed; uniform spacing used")

    # eval() for a deterministic FLOPs probe, but restore the caller's mode:
    # silently leaving a training backbone in eval would freeze BatchNorm in a
    # custom training loop.
    was_training = backbone.training
    backbone.eval()
    try:
        # Reused-module detection: a leaf that executes more than once makes
        # "cumulative FLOPs up to layer L in registration order" meaningless.
        call_counts = _count_leaf_calls(backbone, input_shape, device)
        reused = sorted(name for name, c in call_counts.items() if c > 1)
        if reused:
            shown = ", ".join(reused[:5]) + (", ..." if len(reused) > 5 else "")
            warnings.warn(
                f"backbone reuses module(s) [{shown}] within one forward pass; "
                "per-layer cumulative FLOPs cannot be attributed reliably. "
                "Falling back to a low-confidence uniform estimate "
                "(FlopsEstimate.reliable=False).",
                RuntimeWarning,
                stacklevel=2,
            )
            return _uniform(layer_names, f"reused modules invalidate ordering: {shown}")

        dummy = torch.zeros(input_shape, device=device)
        fca = FlopCountAnalysis(backbone, dummy)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        total = fca.total()
        by_module = fca.by_module()

        if total == 0:
            return _uniform(layer_names, "estimator reported zero total FLOPs")

        # Forward order = order returned by named_modules. Filter to leaves only
        # (modules with no children) to avoid double-counting containers.
        ordered_leaves = [
            name
            for name, mod in backbone.named_modules()
            if name and len(list(mod.children())) == 0
        ]
        leaf_pos = {leaf: i for i, leaf in enumerate(ordered_leaves)}

        # fvcore sometimes attributes FLOPs to a NON-leaf module rather than its
        # leaves (e.g. nn.MultiheadAttention in a ViT block reports its FLOPs on
        # the attention module, with its out_proj leaf showing 0). The leaf walk
        # alone would drop those. Recover each non-leaf module's "self" FLOPs
        # (its own count minus its direct children's) and attribute them to the
        # position where the module finishes (its last descendant leaf). For pure
        # CNNs every non-leaf self-count is 0, so this is a no-op there.
        nonleaf_self: list[tuple[int, int]] = []
        for name, mod in backbone.named_modules():
            children = list(mod.named_children())
            if not name or not children:
                continue
            desc_positions = [leaf_pos[lf] for lf in ordered_leaves if lf.startswith(name + ".")]
            if not desc_positions:
                continue
            own = int(by_module.get(name, 0))
            child_sum = sum(int(by_module.get(f"{name}.{cn}", 0)) for cn, _ in children)
            self_flops = own - child_sum
            if self_flops > 0:
                nonleaf_self.append((max(desc_positions), self_flops))

        # The "end" of an exit at layer L is the last leaf whose name belongs to
        # L (equals L or is nested under it). Anything past that point hasn't run
        # yet when the exit fires.
        notes: list[str] = []
        cumulative: dict[str, float] = {}
        for layer in layer_names:
            last_idx = -1
            for i, leaf in enumerate(ordered_leaves):
                if leaf == layer or leaf.startswith(layer + "."):
                    last_idx = i
            if last_idx < 0:
                # target isn't a leaf or doesn't contain leaves: fall back to
                # uniform spacing for this entry
                n = len(layer_names)
                cumulative[layer] = (list(layer_names).index(layer) + 1) / (n + 1)
                notes.append(f"layer {layer!r} has no leaves; uniform value used")
                continue
            running = sum(int(by_module.get(ordered_leaves[i], 0)) for i in range(last_idx + 1))
            running += sum(flops for pos, flops in nonleaf_self if pos <= last_idx)
            ratio = running / total
            if ratio > 1.05:
                # latent overcount: either fvcore.total() underreports for
                # partially-supported ops, or our leaf walk picked up something it
                # shouldn't. clamp below for safety but surface the discrepancy.
                warnings.warn(
                    f"flops accounting overcount at layer {layer!r}: "
                    f"running/total={ratio:.3f} (> 1.05); clamping to 1.0",
                    RuntimeWarning,
                    stacklevel=2,
                )
                notes.append(f"overcount clamped at {layer!r}")
            cumulative[layer] = min(ratio, 1.0)

        return FlopsEstimate(
            fractions=cumulative,
            method=METHOD_FVCORE,
            reliable=not notes,
            notes=tuple(notes),
        )
    finally:
        backbone.train(was_training)


def per_layer_flops(
    backbone: nn.Module,
    layer_names: Sequence[str],
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
) -> dict[str, float]:
    """Backward-compatible wrapper returning just the fraction mapping.

    Prefer :func:`estimate_layer_flops`, which also reports the method used
    and whether the estimate is reliable.
    """
    return estimate_layer_flops(backbone, layer_names, input_shape).fractions
