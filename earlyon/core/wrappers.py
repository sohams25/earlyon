"""EarlyExitWrapper: wraps any backbone with hook-attached early exits.

Routing happens at inference time. To avoid running layers after a confident
exit fires, the wrapper raises a sentinel exception inside the forward hook
and catches it at the top level. This is the only reliable way to
short-circuit an opaque ``nn.Module.forward`` from inside a hook.
"""

from __future__ import annotations

import threading
from typing import Callable, Iterator, Optional

import torch
import torch.nn as nn

from earlyon.core.flops import per_layer_flops
from earlyon.core.types import EarlyExitConfig, InferenceResult


class _EarlyExitSignal(Exception):
    """Raised inside a forward hook to short-circuit the backbone."""

    def __init__(self, exit_idx: int, prediction: torch.Tensor, confidence: float) -> None:
        super().__init__()
        self.exit_idx = exit_idx
        self.prediction = prediction
        self.confidence = confidence


class EarlyExitWrapper(nn.Module):
    """Wraps a backbone, attaching exit heads at the layers named in config.

    Parameters
    ----------
    backbone:
        Any ``nn.Module``. Must be callable on the input shape used for
        FLOPs accounting (default ``(1, 3, 224, 224)``).
    exit_heads:
        Mapping from ``ExitPoint.name`` to an ``EarlyExitHead`` instance. The
        head must accept the feature tensor produced at ``layer_name``.
    final_classifier:
        Callable mapping the *final* feature tensor to logits. For
        ``torchvision`` ResNets this is ``backbone.fc(backbone.avgpool(...).flatten(1))``;
        for MobileNetV2 it is ``backbone.classifier(...)``. The model factories
        in ``earlyon.models`` build the right closure.
    config:
        ``EarlyExitConfig`` driving routing and loss weights.
    input_shape:
        Used for one-time FLOPs accounting at construction.
    """

    def __init__(
        self,
        backbone: nn.Module,
        exit_heads: dict[str, nn.Module],
        final_classifier: Callable[[torch.Tensor], torch.Tensor],
        config: EarlyExitConfig,
        input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.exit_heads = nn.ModuleDict(exit_heads)
        self._final_classifier = final_classifier
        self.config = config

        # ordered list of (exit_idx, exit_name, layer_name)
        self._exits = [
            (i, ep.name, ep.layer_name) for i, ep in enumerate(config.exit_points)
        ]

        # cumulative FLOPs fraction at each exit layer
        layer_names = [ep.layer_name for ep in config.exit_points]
        self._flops_at: dict[str, float] = per_layer_flops(
            backbone, layer_names, input_shape
        )

        # per-thread caches: a single wrapper instance can be called from
        # multiple threads (DataParallel, inference servers). state lives in
        # threading.local so concurrent training/inference calls don't collide.
        self._tls = threading.local()

        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    # ---------------- public api ----------------

    def forward(
        self, x: torch.Tensor, mode: str = "inference"
    ) -> InferenceResult | list[torch.Tensor]:
        if mode == "training":
            return self._forward_training(x)
        if mode == "inference":
            return self._forward_inference(x)
        raise ValueError(f"mode must be 'training' or 'inference', got {mode!r}")

    def exit_parameters(self) -> Iterator[nn.Parameter]:
        for head in self.exit_heads.values():
            yield from head.parameters()

    def backbone_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.backbone.parameters()

    # ---------------- internals ----------------

    def _register_hooks(self) -> None:
        for exit_idx, exit_name, layer_name in self._exits:
            module = self.backbone.get_submodule(layer_name)
            handle = module.register_forward_hook(
                self._make_hook(exit_idx, exit_name)
            )
            self._hook_handles.append(handle)

    def _make_hook(self, exit_idx: int, exit_name: str):
        def hook(module: nn.Module, inputs, output):
            head = self.exit_heads[exit_name]
            logits = head(output)
            if getattr(self._tls, "inference_mode", False):
                # single-sample routing only (v0.1)
                if logits.size(0) != 1:
                    raise RuntimeError(
                        "EarlyExitWrapper inference requires batch_size=1 "
                        "in v0.1 (got batch_size="
                        f"{logits.size(0)}); see README"
                    )
                temp = max(self.config.temperature, 1e-6)
                probs = torch.softmax(logits / temp, dim=-1)
                confidence = probs.max().item()
                # read thresholds direct from config — the per-call copy was
                # redundant
                threshold = self.config.confidence_thresholds[exit_idx]
                if confidence >= threshold:
                    raise _EarlyExitSignal(exit_idx, logits, confidence)
            else:
                getattr(self._tls, "training_outputs", []).append(logits)
            return output

        return hook

    def _forward_training(self, x: torch.Tensor) -> list[torch.Tensor]:
        self._tls.training_outputs = []
        self._tls.inference_mode = False
        try:
            feats = self.backbone(x)
            final_logits = self._final_classifier(feats)
            return list(self._tls.training_outputs) + [final_logits]
        finally:
            self._tls.training_outputs = []

    def _forward_inference(self, x: torch.Tensor) -> InferenceResult:
        self._tls.inference_mode = True
        try:
            try:
                feats = self.backbone(x)
            except _EarlyExitSignal as sig:
                layer_name = self._exits[sig.exit_idx][2]
                return InferenceResult(
                    prediction=sig.prediction,
                    exit_taken=sig.exit_idx,
                    confidence=sig.confidence,
                    computation_used=self._flops_at[layer_name],
                )
            # no exit triggered — use final classifier (inside outer try so
            # any exception still resets inference_mode)
            final_logits = self._final_classifier(feats)
            temp = max(self.config.temperature, 1e-6)
            probs = torch.softmax(final_logits / temp, dim=-1)
            confidence = probs.max().item()
            return InferenceResult(
                prediction=final_logits,
                exit_taken=-1,
                confidence=confidence,
                computation_used=1.0,
            )
        finally:
            self._tls.inference_mode = False
