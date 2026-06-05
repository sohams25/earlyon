"""Early-exit wrapper for torchvision Vision Transformer (ViT-B/16).

ViT is earlyon's first non-CNN backbone. The encoder is 12 ``EncoderBlock``s,
each emitting a ``(B, 197, 768)`` token sequence (196 patch tokens + 1 CLS).
Exits attach after blocks 3 and 9 — the most even FLOPs spread (~0.34 / ~0.83
of the network) — and use the generalised :class:`EarlyExitHead` with
``pool_tokens="cls"`` to mirror ViT's own class-token readout.

Note: fvcore counts the attention projection matmuls but not the
scaled-dot-product-attention matmuls themselves, so ``computation_used`` is a
slight (few-percent) under-count of true FLOPs; the fractions stay monotonic.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16

from earlyon.core.exit_head import EarlyExitHead
from earlyon.core.types import EarlyExitConfig, ExitPoint
from earlyon.core.wrappers import EarlyExitWrapper
from earlyon.models._common import identity

_VIT_B16_HIDDEN = 768
_VIT_B16_INPUT_SHAPE = (1, 3, 224, 224)  # torchvision vit_b_16 image_size=224
_VIT_B16_EXITS = [
    ExitPoint("e0", "encoder.layers.encoder_layer_3", _VIT_B16_HIDDEN),
    ExitPoint("e1", "encoder.layers.encoder_layer_9", _VIT_B16_HIDDEN),
]


def vit_b_16_ee(num_classes: int, pretrained: bool = True) -> EarlyExitWrapper:
    """Wrap torchvision ``vit_b_16`` with two early exits (after encoder blocks
    3 and 9).

    ``vit_b_16.forward`` already applies the classification head and returns
    ``(B, num_classes)`` logits, so ``final_classifier`` is identity. When
    ``num_classes != 1000`` the final head is replaced with a fresh
    ``nn.Linear(768, num_classes)``.
    """
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = vit_b_16(weights=weights)
    if num_classes != 1000:
        backbone.heads.head = nn.Linear(_VIT_B16_HIDDEN, num_classes)

    heads: dict[str, nn.Module] = {
        ep.name: EarlyExitHead(ep.in_channels, num_classes, hidden_dim=256, pool_tokens="cls")
        for ep in _VIT_B16_EXITS
    }
    cfg = EarlyExitConfig(
        backbone="vit_b_16",
        num_classes=num_classes,
        exit_points=list(_VIT_B16_EXITS),
        confidence_thresholds=[0.85, 0.80],
        loss_weights=[0.2, 0.3, 0.5],
    )
    return EarlyExitWrapper(backbone, heads, identity, cfg, input_shape=_VIT_B16_INPUT_SHAPE)
