"""FLOPs accounting helpers.

Uses fvcore for per-module flops. The non-trivial part is: when an exit point
is a nested module like ``features.3`` (MobileNetV2), we need to sum the flops
of every LEAF op executed before and including ``features.3`` in forward
order, without double-counting the container module.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import torch
import torch.nn as nn


def per_layer_flops(
    backbone: nn.Module,
    layer_names: Sequence[str],
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
) -> dict[str, float]:
    """Return the cumulative FLOPs fraction at each named layer.

    At layer L the value is (FLOPs of every leaf op executed up to and
    including L) / (total backbone FLOPs). The result is monotonic in
    forward order and bounded in [0, 1].
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        n = len(layer_names)
        return {name: (i + 1) / (n + 1) for i, name in enumerate(layer_names)}

    # eval() for a deterministic FLOPs probe, but restore the caller's mode:
    # this runs inside EarlyExitWrapper.__init__, and silently leaving a
    # training backbone in eval would freeze BatchNorm in a custom training loop.
    was_training = backbone.training
    backbone.eval()
    try:
        dummy = torch.zeros(input_shape)
        fca = FlopCountAnalysis(backbone, dummy)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        total = fca.total()
        by_module = fca.by_module()

        if total == 0:
            n = len(layer_names)
            return {name: (i + 1) / (n + 1) for i, name in enumerate(layer_names)}

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
            cumulative[layer] = min(ratio, 1.0)

        return cumulative
    finally:
        backbone.train(was_training)
