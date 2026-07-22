"""Wrapper invariants and routing behavior."""

import warnings

import pytest
import torch

from earlyon.core.exit_head import EarlyExitHead
from earlyon.core.types import EarlyExitConfig, ExitPoint
from earlyon.core.wrappers import EarlyExitWrapper
from tests.fixtures.tiny_models import STAGE_CHANNELS, TinyBackbone


def _build(thresholds=(0.99, 0.99, 0.99)):
    backbone = TinyBackbone(num_classes=10)
    exits = [
        ExitPoint("e0", "stage1", STAGE_CHANNELS["stage1"]),
        ExitPoint("e1", "stage2", STAGE_CHANNELS["stage2"]),
        ExitPoint("e2", "stage3", STAGE_CHANNELS["stage3"]),
    ]
    heads = {ep.name: EarlyExitHead(ep.in_channels, 10) for ep in exits}
    cfg = EarlyExitConfig(
        backbone="tiny",
        num_classes=10,
        exit_points=exits,
        confidence_thresholds=list(thresholds),
    )

    def final_classifier(feats):
        # feats is already the backbone output (logits) since TinyBackbone.fc
        # runs internally. We pass through identity.
        return feats

    wrapper = EarlyExitWrapper(
        backbone=backbone,
        exit_heads=heads,
        final_classifier=final_classifier,
        config=cfg,
        input_shape=(1, 3, 32, 32),
    )
    return wrapper, backbone


def test_training_mode_returns_all_exits():
    wrapper, _ = _build()
    x = torch.randn(4, 3, 32, 32)
    outputs = wrapper(x, mode="training")
    assert len(outputs) == 4  # 3 exits + final
    for o in outputs:
        assert o.shape == (4, 10)


def test_inference_returns_inference_result():
    wrapper, _ = _build()
    x = torch.randn(1, 3, 32, 32)
    result = wrapper(x, mode="inference")
    assert result.prediction.shape == (1, 10)
    assert -1 <= result.exit_taken < 3
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 < result.computation_used <= 1.0


@pytest.mark.parametrize("thresholds", [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])
def test_inference_prediction_is_grad_free_without_external_no_grad(thresholds):
    """Regression for the autograd-graph leak: calling inference WITHOUT an
    external no_grad/inference_mode context (exactly the documented quick-start
    pattern) must still return a graph-free prediction and emit no
    requires_grad-to-scalar UserWarning. Covers both the early-exit path
    (thresholds=0) and the final-classifier fall-through (thresholds=1)."""
    wrapper, _ = _build(thresholds=thresholds)
    wrapper.eval()
    x = torch.randn(1, 3, 32, 32)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = wrapper(x, mode="inference")  # deliberately no no_grad()
    assert result.prediction.requires_grad is False
    assert result.prediction.grad_fn is None
    grad_warnings = [w for w in record if "requires_grad" in str(w.message)]
    assert not grad_warnings, f"unexpected requires_grad warning: {grad_warnings}"


def test_batched_inference_prediction_is_grad_free_without_external_no_grad():
    """Same grad-free guarantee for the batched routing path."""
    wrapper = _build(thresholds=(0.0, 0.0, 0.0))[0]
    wrapper.eval()
    result = wrapper.forward_inference_batched(torch.randn(4, 3, 32, 32))
    assert result.predictions.requires_grad is False
    assert result.predictions.grad_fn is None
    assert result.per_sample_confidence.requires_grad is False


@pytest.mark.parametrize("bad_temp", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_temperature_does_not_poison_softmax(bad_temp):
    """Config validation rejects non-finite temperatures at the trust
    boundaries, but a direct post-hoc mutation of config.temperatures bypasses
    it; the runtime guard must still fall back to 1.0 so routing stays
    well-defined and confidence is finite."""
    import math

    wrapper, _ = _build(thresholds=(1.0, 1.0, 1.0))
    wrapper.eval()
    for head in wrapper.config.temperatures:
        wrapper.config.temperatures[head] = bad_temp  # bypasses validate()
    result = wrapper(torch.randn(1, 3, 32, 32), mode="inference")
    assert math.isfinite(result.confidence)
    assert torch.isfinite(result.prediction).all()


def test_disabled_exit_cannot_fire_even_at_saturated_confidence():
    """The P0 enablement invariant: an exit with enabled_exits[i]=False must
    never fire, even when its softmax confidence is exactly 1.0 (float32
    saturation) and its threshold is 1.0 — the numerical edge case the old
    'threshold sentinel means disabled' convention got wrong."""
    wrapper, _ = _build(thresholds=(1.0, 1.0, 1.0))
    # saturate exit e0: huge constant logit -> softmax max exactly 1.0
    with torch.no_grad():
        wrapper.exit_heads["e0"].classifier[-1].bias.zero_()
        wrapper.exit_heads["e0"].classifier[-1].bias[0] = 1e3
    wrapper.eval()
    x = torch.randn(1, 3, 32, 32)

    # sanity: with the exit enabled and threshold 1.0, saturation DOES fire
    wrapper.config.enabled_exits = [True, False, False]
    assert wrapper(x, mode="inference").exit_taken == 0

    # disabled: must fall through to the final classifier
    wrapper.config.enabled_exits = [False, False, False]
    result = wrapper(x, mode="inference")
    assert result.exit_taken == -1
    assert result.estimated_backbone_flops_fraction == 1.0

    # batched path honors the same invariant
    batched = wrapper.forward_inference_batched(torch.randn(4, 3, 32, 32))
    assert batched.exit_taken == -1


def test_disabled_exit_cannot_fire_at_entropy_exactly_zero():
    """Same invariant for the entropy policy: a saturated head has entropy
    exactly 0.0, which meets any threshold >= 0; disabling the exit must win."""
    backbone = TinyBackbone(num_classes=10)
    exits = [
        ExitPoint("e0", "stage1", STAGE_CHANNELS["stage1"]),
        ExitPoint("e1", "stage2", STAGE_CHANNELS["stage2"]),
    ]
    heads = {ep.name: EarlyExitHead(ep.in_channels, 10) for ep in exits}
    cfg = EarlyExitConfig(
        backbone="tiny",
        num_classes=10,
        exit_points=exits,
        routing_policy="entropy",
        entropy_thresholds=[0.0, 0.0],
    )
    wrapper = EarlyExitWrapper(backbone, heads, lambda t: t, cfg, input_shape=(1, 3, 32, 32))
    with torch.no_grad():
        wrapper.exit_heads["e0"].classifier[-1].bias.zero_()
        wrapper.exit_heads["e0"].classifier[-1].bias[0] = 1e3
    wrapper.eval()
    x = torch.randn(1, 3, 32, 32)

    wrapper.config.enabled_exits = [True, False]
    assert wrapper(x, mode="inference").exit_taken == 0  # entropy 0.0 <= 0.0 fires

    wrapper.config.enabled_exits = [False, False]
    assert wrapper(x, mode="inference").exit_taken == -1


def test_disabled_exit_still_produces_training_logits():
    """Enablement is a routing concept: training mode must return logits for
    every head, disabled or not, so a disabled head keeps learning."""
    wrapper, _ = _build()
    wrapper.config.enabled_exits = [False, True, False]
    outputs = wrapper(torch.randn(2, 3, 32, 32), mode="training")
    assert len(outputs) == 4  # all 3 exits + final, regardless of enablement


def test_per_head_temperature_is_applied_at_the_right_exit():
    """Each exit must use its own temperature. A very high temperature flattens
    exit e0's softmax below the threshold (no fire); resetting only e0's
    temperature to 1.0 makes the same input fire at e0 again."""
    wrapper, _ = _build(thresholds=(0.9, 0.9, 0.9))
    with torch.no_grad():
        wrapper.exit_heads["e0"].classifier[-1].bias.zero_()
        wrapper.exit_heads["e0"].classifier[-1].bias[0] = 20.0  # confident but not saturated
        # make later heads and final produce flat logits so nothing else fires
        wrapper.exit_heads["e1"].classifier[-1].weight.zero_()
        wrapper.exit_heads["e1"].classifier[-1].bias.zero_()
        wrapper.exit_heads["e2"].classifier[-1].weight.zero_()
        wrapper.exit_heads["e2"].classifier[-1].bias.zero_()
    wrapper.eval()
    x = torch.randn(1, 3, 32, 32)

    wrapper.config.temperatures = {"e0": 1.0, "e1": 1.0, "e2": 1.0, "final": 1.0}
    assert wrapper(x, mode="inference").exit_taken == 0

    # flatten ONLY e0 via its per-head temperature; other heads unchanged
    wrapper.config.temperatures = {"e0": 100.0, "e1": 1.0, "e2": 1.0, "final": 1.0}
    result = wrapper(x, mode="inference")
    assert result.exit_taken == -1, "e0's own temperature must govern e0's routing"


def test_wrapper_rejects_mismatched_exit_heads():
    """Heads must exactly match the configured exit points."""
    backbone = TinyBackbone(num_classes=10)
    exits = [ExitPoint("e0", "stage1", STAGE_CHANNELS["stage1"])]
    cfg = EarlyExitConfig(backbone="tiny", num_classes=10, exit_points=exits)
    with pytest.raises(ValueError, match="exit_heads"):
        EarlyExitWrapper(backbone, {}, lambda t: t, cfg, input_shape=(1, 3, 32, 32))
    heads = {
        "e0": EarlyExitHead(STAGE_CHANNELS["stage1"], 10),
        "extra": EarlyExitHead(STAGE_CHANNELS["stage2"], 10),
    }
    with pytest.raises(ValueError, match="exit_heads"):
        EarlyExitWrapper(backbone, heads, lambda t: t, cfg, input_shape=(1, 3, 32, 32))


def test_safe_temperature_clamps_to_positive_finite():
    """Unit-level guarantees of the temperature guard."""
    from earlyon.core.wrappers import _safe_temperature

    assert _safe_temperature(float("nan")) == 1.0
    assert _safe_temperature(float("inf")) == 1.0
    assert _safe_temperature(float("-inf")) == 1.0
    # non-positive values fall back to the no-op temperature 1.0 — NOT a tiny
    # positive clamp, which would produce an artificially razor-sharp softmax
    assert _safe_temperature(0.0) == 1.0
    assert _safe_temperature(-5.0) == 1.0
    assert _safe_temperature(2.5) == 2.5  # finite positive passes through


def test_thresholds_one_means_no_early_exit():
    """The core invariant: with thresholds=1.0, no exit can fire because
    softmax(logits).max() is strictly < 1. The wrapper must reach the final
    classifier and return the backbone output unchanged."""
    wrapper, backbone = _build(thresholds=(1.0, 1.0, 1.0))
    wrapper.eval()
    backbone.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        result = wrapper(x, mode="inference")
        direct = backbone(x)
    assert result.exit_taken == -1
    assert result.computation_used == 1.0
    assert torch.allclose(result.prediction, direct, atol=1e-5)


def test_zero_threshold_exits_immediately():
    """With threshold=0, the first exit always fires."""
    wrapper, _ = _build(thresholds=(0.0, 0.0, 0.0))
    wrapper.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        result = wrapper(x, mode="inference")
    assert result.exit_taken == 0
    assert result.computation_used < 1.0


def test_inference_rejects_batch_gt_1():
    wrapper, _ = _build(thresholds=(0.0, 0.0, 0.0))
    x = torch.randn(2, 3, 32, 32)
    with pytest.raises(RuntimeError, match="batch_size=1"):
        wrapper(x, mode="inference")


def test_exit_parameters_excludes_backbone():
    wrapper, _ = _build()
    exit_param_ids = {id(p) for p in wrapper.exit_parameters()}
    backbone_param_ids = {id(p) for p in wrapper.backbone_parameters()}
    assert exit_param_ids.isdisjoint(backbone_param_ids)
    assert len(exit_param_ids) > 0


def test_computation_used_monotonic_across_exits():
    """Later exits must report higher cumulative FLOPs than earlier ones."""
    wrapper, _ = _build(thresholds=(0.0, 0.0, 0.0))
    wrapper.eval()
    flops_at = wrapper._flops_at
    values = [flops_at[ep.layer_name] for ep in wrapper.config.exit_points]
    assert values == sorted(values)
    assert all(0 < v <= 1 for v in values)


def test_training_mode_repeated_calls_do_not_accumulate():
    wrapper, _ = _build()
    x = torch.randn(2, 3, 32, 32)
    out1 = wrapper(x, mode="training")
    out2 = wrapper(x, mode="training")
    assert len(out1) == 4 and len(out2) == 4


def test_inference_mode_reset_after_final_classifier_error():
    """If _final_classifier raises, the wrapper must still reset inference
    mode so subsequent training calls behave correctly."""
    wrapper, _ = _build(thresholds=(1.0, 1.0, 1.0))

    def bad_classifier(x):
        raise RuntimeError("simulated downstream error")

    wrapper._final_classifier = bad_classifier
    with pytest.raises(RuntimeError, match="simulated"):
        wrapper(torch.randn(1, 3, 32, 32), mode="inference")
    # inference flag must be off — training mode should now collect outputs
    wrapper._final_classifier = lambda v: v
    out = wrapper(torch.randn(2, 3, 32, 32), mode="training")
    assert len(out) == 4


def test_standalone_backbone_call_does_not_fire_hooks():
    """When the wrapper's backbone is called directly (e.g. stage 1 training),
    exit heads must not fire. Otherwise a backbone-only `.to(device)` leaves
    the heads on cpu and causes a device-mismatch error.
    """
    wrapper, backbone = _build()
    wrapper.eval()
    x = torch.randn(2, 3, 32, 32)
    # Move only the backbone (mimics stage1_train_backbone)
    out_direct = backbone(x)
    # The hook must have been a no-op: no exception, no head invocation
    assert out_direct.shape == (2, 10)


def test_batched_inference_routes_all_or_none():
    """Per-batch routing: if all samples meet threshold, batch exits early."""
    from earlyon.core.types import BatchedInferenceResult

    wrapper, _ = _build(thresholds=(0.0, 0.0, 0.0))  # always exit at e0
    wrapper.eval()
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        result = wrapper.forward_inference_batched(x)
    assert isinstance(result, BatchedInferenceResult)
    assert result.predictions.shape == (4, 10)
    assert result.exit_taken == 0
    assert result.per_sample_confidence.shape == (4,)
    assert result.computation_used < 1.0


def test_batched_inference_high_threshold_falls_through_to_final():
    wrapper, _ = _build(thresholds=(1.0, 1.0, 1.0))
    wrapper.eval()
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        result = wrapper.forward_inference_batched(x)
    assert result.exit_taken == -1
    assert result.computation_used == 1.0


def test_batched_inference_empty_batch_raises():
    wrapper, _ = _build()
    with pytest.raises(ValueError, match="empty batch"):
        wrapper.forward_inference_batched(torch.zeros(0, 3, 32, 32))


def _build_entropy(entropy_thresholds=(0.1, 0.1, 0.1)):
    backbone = TinyBackbone(num_classes=10)
    exits = [
        ExitPoint("e0", "stage1", STAGE_CHANNELS["stage1"]),
        ExitPoint("e1", "stage2", STAGE_CHANNELS["stage2"]),
        ExitPoint("e2", "stage3", STAGE_CHANNELS["stage3"]),
    ]
    heads = {ep.name: EarlyExitHead(ep.in_channels, 10) for ep in exits}
    cfg = EarlyExitConfig(
        backbone="tiny",
        num_classes=10,
        exit_points=exits,
        routing_policy="entropy",
        entropy_thresholds=list(entropy_thresholds),
    )
    return EarlyExitWrapper(backbone, heads, lambda x: x, cfg, input_shape=(1, 3, 32, 32))


def test_entropy_routing_exits_on_high_threshold():
    """Wide entropy threshold (= accept high uncertainty) means even untrained
    heads exit early. With threshold = log(10) ≈ 2.30 every distribution
    qualifies and exit 0 fires."""
    import math

    wrapper = _build_entropy(entropy_thresholds=(math.log(10) + 0.01,) * 3)
    wrapper.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        result = wrapper(x, mode="inference")
    assert result.exit_taken == 0
    assert result.computation_used < 1.0


def test_entropy_routing_does_not_exit_on_tight_threshold():
    """Threshold near zero rejects every realistic distribution → no exit fires."""
    wrapper = _build_entropy(entropy_thresholds=(1e-6, 1e-6, 1e-6))
    wrapper.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        result = wrapper(x, mode="inference")
    assert result.exit_taken == -1
    assert result.computation_used == 1.0


def test_entropy_routing_preserves_inference_result_shape():
    """Swapping routing policy must not change the InferenceResult contract."""
    wrapper = _build_entropy(entropy_thresholds=(0.5, 0.5, 0.5))
    wrapper.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        result = wrapper(x, mode="inference")
    assert result.prediction.shape == (1, 10)
    assert isinstance(result.exit_taken, int)
    assert isinstance(result.confidence, float)
    assert isinstance(result.computation_used, float)


def test_entropy_routing_batched_inference():
    """Batched routing must also work with entropy policy: all samples in a
    batch exit together when every sample's entropy is below threshold."""
    import math

    wrapper = _build_entropy(entropy_thresholds=(math.log(10) + 0.01,) * 3)
    wrapper.eval()
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        result = wrapper.forward_inference_batched(x)
    assert result.exit_taken == 0
    assert result.predictions.shape == (4, 10)


def test_training_mode_on_fresh_thread_does_not_drop_outputs():
    """The threading.local must have its schema set before the hook fires.
    Without _init_tls, a fresh thread sees a missing training_outputs
    attribute and the getattr default '[]' is discarded — silently
    producing only the final classifier logit and a wrong loss.
    """
    import threading

    wrapper, _ = _build()
    x = torch.randn(2, 3, 32, 32)
    captured = {}

    def run():
        captured["outputs"] = wrapper(x, mode="training")

    t = threading.Thread(target=run)
    t.start()
    t.join()

    assert "outputs" in captured
    assert len(captured["outputs"]) == 4  # 3 exits + final
    for o in captured["outputs"]:
        assert o.shape == (2, 10)
