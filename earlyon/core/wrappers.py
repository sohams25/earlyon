"""EarlyExitWrapper: wraps any backbone with hook-attached early exits.

Routing happens at inference time. To avoid running layers after a confident
exit fires, the wrapper raises a sentinel exception inside the forward hook
and catches it at the top level. This is the only reliable way to
short-circuit an opaque ``nn.Module.forward`` from inside a hook.
"""

from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Optional

import torch
import torch.nn as nn

from earlyon.core.flops import per_layer_flops
from earlyon.core.types import EarlyExitConfig, InferenceResult

if TYPE_CHECKING:
    from earlyon.core.types import BatchedInferenceResult


def _safe_temperature(temperature: float) -> float:
    """Clamp the softmax temperature to a strictly positive, finite value.

    A NaN/Inf temperature (e.g. from a diverged ``fit_temperature`` on degenerate
    logits) would otherwise poison every softmax: ``max(nan, 1e-6)`` returns
    ``nan`` because Python ``max`` keeps its first argument on a False compare,
    and ``softmax(logits / nan)`` is silently all-NaN. Falling back to 1.0
    (the no-op temperature) keeps routing well-defined.
    """
    if not math.isfinite(temperature):
        return 1.0
    return max(temperature, 1e-6)


def _is_compiling() -> bool:
    """True if running inside torch.compile / dynamo tracing."""
    try:
        import torch._dynamo as dynamo

        return dynamo.is_compiling()
    except Exception:
        return False


class _EarlyExitSignal(Exception):
    """Raised inside a forward hook to short-circuit the backbone."""

    def __init__(
        self,
        exit_idx: int,
        prediction: torch.Tensor,
        confidence: float,
        per_sample_confidence: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.exit_idx = exit_idx
        self.prediction = prediction
        self.confidence = confidence
        self.per_sample_confidence = per_sample_confidence


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
        exit_heads: Mapping[str, nn.Module],
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
        self._exits = [(i, ep.name, ep.layer_name) for i, ep in enumerate(config.exit_points)]

        # cumulative FLOPs fraction at each exit layer
        layer_names = [ep.layer_name for ep in config.exit_points]
        self._flops_at: dict[str, float] = per_layer_flops(backbone, layer_names, input_shape)

        # per-thread caches: a single wrapper instance can be called from
        # multiple threads (DataParallel, inference servers). state lives in
        # threading.local so concurrent training/inference calls don't collide.
        #
        # CAVEAT: this state is per-thread, not per-call-frame. Re-entrant use of
        # the same wrapper on one thread — i.e. a custom ``backbone`` or
        # ``final_classifier`` that itself calls back into this wrapper's
        # forward — is unsupported: the inner call's ``finally`` resets the
        # inference/training flags while the outer call is still running. No
        # shipped backbone or factory does this (final_classifier is identity),
        # so it cannot occur in normal use.
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

    def _init_tls(self) -> None:
        """Ensure per-thread state has the expected schema. Without this, a
        wrapper called for the first time on a fresh thread (e.g. a
        DataParallel replica) would silently no-op the hook because
        ``training_outputs`` and ``in_training`` are missing on this
        thread's ``threading.local`` and the getattr default discards
        appended logits."""
        if not hasattr(self._tls, "training_outputs"):
            self._tls.training_outputs = []
        if not hasattr(self._tls, "in_training"):
            self._tls.in_training = False
        if not hasattr(self._tls, "inference_mode"):
            self._tls.inference_mode = False
        if not hasattr(self._tls, "batched_mode"):
            self._tls.batched_mode = False

    def forward_inference_batched(self, x: torch.Tensor) -> "BatchedInferenceResult":
        """Conservative per-batch routing for batch_size > 1.

        All samples in the batch exit together at the earliest layer where
        EVERY sample's confidence is above its threshold. The hardest sample
        dictates the exit point. This gives real speedups when batches are
        homogeneously easy, no speedup when even one sample is hard.

        This is the v0.2 batched mode. For true per-sample routing inside a
        batch (with masked execution), see the v0.3 roadmap.
        """
        from earlyon.core.types import BatchedInferenceResult

        if x.size(0) < 1:
            raise ValueError("empty batch")

        self._init_tls()
        self._tls.inference_mode = True
        self._tls.batched_mode = True
        try:
            # inference_mode() keeps the returned logits/confidences free of an
            # autograd graph; without it a caller that does not wrap the call in
            # no_grad (e.g. the README quick-start) retains a graph per call and
            # leaks memory in a server loop.
            with torch.inference_mode():
                try:
                    feats = self.backbone(x)
                except _EarlyExitSignal as sig:
                    layer_name = self._exits[sig.exit_idx][2]
                    # invariant: a batched-mode raise always carries this tensor.
                    # explicit raise (not assert) so it survives `python -O`.
                    if sig.per_sample_confidence is None:
                        raise RuntimeError(
                            "batched early-exit signal must carry per_sample_confidence"
                        )
                    return BatchedInferenceResult(
                        predictions=sig.prediction,
                        exit_taken=sig.exit_idx,
                        per_sample_confidence=sig.per_sample_confidence,
                        computation_used=self._flops_at[layer_name],
                    )
                # no exit fired
                final_logits = self._final_classifier(feats)
                temp = _safe_temperature(self.config.temperature)
                probs = torch.softmax(final_logits / temp, dim=-1)
                per_sample_conf = probs.max(dim=-1).values
                return BatchedInferenceResult(
                    predictions=final_logits,
                    exit_taken=-1,
                    per_sample_confidence=per_sample_conf,
                    computation_used=1.0,
                )
        finally:
            self._tls.inference_mode = False
            self._tls.batched_mode = False

    # ---------------- internals ----------------

    def _register_hooks(self) -> None:
        for exit_idx, exit_name, layer_name in self._exits:
            module = self.backbone.get_submodule(layer_name)
            handle = module.register_forward_hook(self._make_hook(exit_idx, exit_name))
            self._hook_handles.append(handle)

    def _make_hook(
        self, exit_idx: int, exit_name: str
    ) -> Callable[[nn.Module, Any, torch.Tensor], torch.Tensor]:
        def hook(module: nn.Module, inputs: Any, output: torch.Tensor) -> torch.Tensor:
            # torch.compile / dynamo: dynamic control flow in this hook will
            # cause silent eager fallback, killing the speedup. detect and
            # bail with a clear error so the user fixes it explicitly.
            in_inference = getattr(self._tls, "inference_mode", False)
            in_training = getattr(self._tls, "in_training", False)
            # backbone called standalone (e.g. stage1_train_backbone passes
            # only the backbone): hooks are no-ops. avoids device mismatch
            # when only the backbone (not the wrapper) has been moved to gpu.
            if not (in_inference or in_training):
                return output

            if _is_compiling():
                raise RuntimeError(
                    "EarlyExitWrapper inference is incompatible with "
                    "torch.compile (dynamic control flow). compile the "
                    "raw backbone instead, or wrap the trained backbone "
                    "without exit heads for compiled inference."
                )

            head = self.exit_heads[exit_name]
            logits = head(output)
            if in_training:
                getattr(self._tls, "training_outputs", []).append(logits)
                return output

            temp = _safe_temperature(self.config.temperature)
            probs = torch.softmax(logits / temp, dim=-1)
            policy = self.config.routing_policy

            if getattr(self._tls, "batched_mode", False):
                per_sample_conf = probs.max(dim=-1).values
                if policy == "confidence":
                    threshold = self.config.confidence_thresholds[exit_idx]
                    fires = (per_sample_conf >= threshold).all()
                else:  # entropy
                    # per-sample entropy: -sum p log p (nats), clamp for log(0)
                    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
                    threshold = self.config.entropy_thresholds[exit_idx]
                    fires = (entropy <= threshold).all()
                if fires:
                    raise _EarlyExitSignal(
                        exit_idx,
                        logits,
                        per_sample_conf.min().item(),
                        per_sample_confidence=per_sample_conf,
                    )
            else:
                if logits.size(0) != 1:
                    raise RuntimeError(
                        "EarlyExitWrapper single-sample inference requires "
                        f"batch_size=1 (got {logits.size(0)}); use "
                        "forward_inference_batched(x) for batched routing"
                    )
                # .item() (not float(...)) so no UserWarning fires even when the
                # caller wraps inference in a weaker torch.no_grad() context.
                confidence = probs.max().item()
                if policy == "confidence":
                    threshold = self.config.confidence_thresholds[exit_idx]
                    if confidence >= threshold:
                        raise _EarlyExitSignal(exit_idx, logits, confidence)
                else:  # entropy
                    # distinct name from the batched-branch ``entropy`` Tensor
                    # above so the closure scope stays single-typed (float).
                    entropy_scalar = (-(probs * probs.clamp_min(1e-12).log()).sum()).item()
                    threshold = self.config.entropy_thresholds[exit_idx]
                    if entropy_scalar <= threshold:
                        raise _EarlyExitSignal(exit_idx, logits, confidence)
            return output

        return hook

    def _forward_training(self, x: torch.Tensor) -> list[torch.Tensor]:
        self._init_tls()
        self._tls.training_outputs = []
        self._tls.inference_mode = False
        self._tls.in_training = True
        try:
            feats = self.backbone(x)
            final_logits = self._final_classifier(feats)
            return list(self._tls.training_outputs) + [final_logits]
        finally:
            self._tls.training_outputs = []
            self._tls.in_training = False

    def _forward_inference(self, x: torch.Tensor) -> InferenceResult:
        self._init_tls()
        self._tls.inference_mode = True
        try:
            # see forward_inference_batched: inference_mode() prevents the
            # returned prediction from carrying an autograd graph.
            with torch.inference_mode():
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
                temp = _safe_temperature(self.config.temperature)
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
