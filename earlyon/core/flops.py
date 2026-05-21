"""FLOPs accounting helpers. Uses fvcore but degrades gracefully."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def per_layer_flops(
    backbone: nn.Module,
    layer_names: Sequence[str],
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
) -> dict[str, float]:
    """Return cumulative FLOPs fraction at each named layer.

    Cumulative meaning: at layer L, the value is (FLOPs of layers up to and
    including L) / (total backbone FLOPs). Used to populate
    ``InferenceResult.computation_used``.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        # fallback: uniform spacing
        n = len(layer_names)
        return {name: (i + 1) / (n + 1) for i, name in enumerate(layer_names)}

    backbone.eval()
    dummy = torch.zeros(input_shape)

    # capture per-module flops
    fca = FlopCountAnalysis(backbone, dummy)
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    total = fca.total()
    by_module = fca.by_module()

    if total == 0:
        n = len(layer_names)
        return {name: (i + 1) / (n + 1) for i, name in enumerate(layer_names)}

    # cumulative walk: sum flops of all named-children encountered up to (and
    # including) each target layer in declaration order
    cumulative: dict[str, float] = {}
    running = 0
    ordered_modules = [n for n, _ in backbone.named_modules() if n]
    target_set = set(layer_names)
    for mod_name in ordered_modules:
        # only count top-level-ish modules: those whose name has no dot
        # OR matches one of the layer_names (so we credit it correctly)
        if "." not in mod_name or mod_name in target_set:
            running += int(by_module.get(mod_name, 0))
        if mod_name in target_set:
            cumulative[mod_name] = running / total

    # fill any missing target with uniform fallback
    for i, name in enumerate(layer_names):
        cumulative.setdefault(name, (i + 1) / (len(layer_names) + 1))
    return cumulative
