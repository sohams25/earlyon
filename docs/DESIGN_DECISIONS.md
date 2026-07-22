# Design Decisions

Why earlyon works the way it does. Each section states the decision, the
alternatives we rejected, and the evidence. If you disagree with one of these,
open an issue; the ones marked *revisitable* are held loosely.

## Two-Stage Training as the default

**Decision:** train the backbone as a normal classifier first, then freeze it
(parameters *and* BatchNorm running stats) and train only the exit heads.
Joint end-to-end training ships too, but as the opt-in
(`joint_train_backbone_and_exits`), not the default.

**Why:**

- Your training recipe survives. Stage 1 is literally your existing loop;
  earlyon isn't involved. Any pretrained checkpoint becomes stage-1 output for
  free, which is how the README benchmarks start from ImageNet weights.
- No gradient conflict. With a weighted multi-exit loss, early-exit gradients
  pull shallow layers toward features that help a small linear head now, at
  the expense of features the deep layers need. Freezing the backbone removes
  the tug-of-war entirely.
- It's cheap. The heads are ~5–13K parameters each. Stage 2 converges in a
  handful of epochs on a laptop GPU because the frozen backbone's activations
  are the only forward cost and nothing deep needs gradients.
- BatchNorm freezing matters. If BN running stats keep updating while
  "frozen", the backbone's behavior drifts under the exit-head training
  distribution and stage-1 accuracy quietly degrades. We freeze both, and the
  tests pin this behavior.

**Cost:** two-stage leaves some accuracy on the table versus joint training,
because the backbone never learns features that serve the early heads. That's
the documented tradeoff; when you have the budget and want peak numbers, use
joint. *Revisitable* if evidence shows joint training holds up well enough to
be the default.

## Greedy per-exit threshold calibration

**Decision:** `calibrate_thresholds` sweeps a fixed threshold grid one exit at
a time, keeping the most aggressive value that holds validation accuracy within
the user's target drop (e.g. 1%).

**Why greedy instead of a joint search:** with 3 exits and a ~10-point grid, a
joint sweep is ~1000 full validation evaluations; greedy is ~30. The exits are
sequential, so a sample that leaves at exit 0 never reaches exit 1: earlier
thresholds shape the population later exits see, and sweeping in network order
respects that dependency. In practice the greedy solution sits at or near the
joint optimum because the per-exit accuracy/coverage curves are monotone in the
threshold.

**Why a user-facing accuracy budget:** "as fast as possible within a 1% drop"
is the question practitioners ask. Exposing raw thresholds as the primary knob
forces users to re-derive calibration themselves; exposing an accuracy budget
makes the tradeoff explicit and testable.

## Compute budgets as a calibration objective, not a routing policy

**Decision:** the compute-budget feature is
`calibrate_thresholds_for_budget`, a second objective for the same greedy
threshold search, rather than a third `routing_policy`.

**Why:** a per-sample runtime budget is ill-posed in earlyon's setting. At
inference time a single sample either meets an exit's criterion or it
doesn't; "spend at most 80% FLOPs on this image" can only be enforced by
forcing an exit regardless of confidence, which silently trades accuracy in
exactly the way the library refuses to do. What deployments actually specify
is an *average* cost ("this model gets X% of the GPU"), and averages are a
property of thresholds over a data distribution, which makes them a
calibration-time quantity. So budget reuses the routing machinery unchanged:
confidence and entropy stay the only runtime policies, and the budget search
aims the same per-exit grid at a different constraint (compute within target,
maximize accuracy) instead of (accuracy drop within target, minimize
compute).

**Honesty requirement:** a budget can be unattainable. If the earliest exit
sits at 40% of FLOPs, no thresholds reach a 0.2 target. The search warns and
returns `budget_met=False` with the least-compute configuration found, and
because it is measured on the calibration set, the guarantee is average-case
over that distribution. The README's limitations section says both out loud.

## Temperature scaling before calibration — one temperature per head

**Decision:** optionally fit a temperature (Guo et al. 2017) on held-out data
before threshold calibration — one **per head**: every exit head and the
final classifier get their own fitted scalar
(`EarlyExitConfig.temperatures`).

**Why per-head:** each exit head is a different classifier reading different
features, with its own miscalibration; a shallow head is typically far more
over-confident than the final classifier. Earlier versions fitted a single
temperature from the final classifier's logits and reused it for every exit
— defensible for none of them. The fit (`fit_head_temperatures`) runs on
cached logits from one pass, and each `TemperatureFitResult` carries a
convergence/fallback status: a diverged LBFGS falls back to 1.0 (no
calibration) with a warning, never to an artificially sharp value.

**Why temperature scaling at all:** modern CNN heads are systematically
over-confident; softmax maxima cluster near 1.0 whether the prediction is
right or wrong. Thresholding an uncalibrated confidence wastes most of the
[0, 1] range and makes the greedy sweep's grid resolution meaningless. One
scalar per head fixes the ranking cheaply without changing argmax
predictions, so accuracy is untouched by construction. We chose temperature
scaling over Platt/isotonic because it's the simplest method with a strong
published track record on exactly this failure mode.

## Explicit exit enablement instead of threshold sentinels

**Decision:** whether an exit may fire is a boolean
(`EarlyExitConfig.enabled_exits`), not a magic threshold value.

**Why:** the old convention — confidence threshold `1.0` (or entropy `0.0`)
means "disabled" — was numerically unsound: the router fires on
`confidence >= threshold`, and a float32-saturated softmax produces
confidence exactly 1.0, so a "disabled" exit could still fire. Explicit
booleans make disablement absolute (the head isn't even evaluated at
inference, saving its overhead too), keep thresholds meaning only
thresholds, and let calibration state its decision (`enabled_exits` on the
result) instead of encoding it. v1 checkpoints migrate their sentinels to
explicit flags on load, with a warning. Disabled heads still produce logits
in training mode — enablement is a routing concept, not a training one.

## `estimated_backbone_flops_fraction`, not "computation used"

**Decision:** the compute number every result reports is named as what it is
— a static *estimate* of the backbone FLOPs fraction — with the estimator's
provenance attached (`wrapper.flops_estimate`: method, reliability, notes).

**Why:** the old name `computation_used` read like a measurement of the
inference that just ran. It never was: it is a one-time fvcore analysis of
the backbone, excludes the exit heads and all routing overhead, and assumes
each leaf module runs once in registration order. Backbones that reuse
modules break that assumption — the estimator now detects reuse and degrades
to a warned, low-confidence uniform fallback instead of a precise-looking
wrong number. The analysis is lazy (first inference) so constructing large
models (ViT) stays fast. `computation_used` survives as a read alias.

## Entropy routing alongside confidence

**Decision:** ship two routing policies: max-softmax confidence (default) and
entropy.

**Why both:** max-confidence only looks at the top class; entropy sees the
whole distribution. A prediction split 0.49/0.48 between two classes and one
split 0.49/0.05/... have the same confidence but very different certainty, and
entropy separates them. Entropy is the standard alternative in the early-exit
literature, costs one extra reduction, and gives users an A/B lever without
leaving the library. Confidence stays the default because it's cheaper to
explain and the CIFAR-10 results between the two are close.

**Consequence for the API:** calibration and serialization are policy-aware.
`calibrate_thresholds` calibrates whichever list the active policy reads, and
`save_wrapper`/`load_wrapper` round-trip the policy plus both threshold lists,
so a calibrated entropy model reloads as an entropy model. Getting this wrong
silently (a calibrated model reloading with the other policy's thresholds) was
judged the worst failure mode, so tests pin the round-trip.

## Forward hooks + a sentinel exception for routing

**Decision:** exit heads attach via `register_forward_hook`, and at inference
time a hook that meets its exit criterion raises a private sentinel exception
that the wrapper catches.

**Why:** the alternative is rewriting each backbone's `forward` to interleave
exit checks, which means one fork per architecture and torchvision version.
Ending that per-paper forking is the reason this library exists. Hooks keep
the backbone byte-identical (weights load unchanged, `custom_ee` can wrap
arbitrary models), but a hook cannot `return` early out of its parent module;
raising is the only reliable way to stop downstream layers from running. The
exception is a private type, raised and caught entirely inside one frame of
library code, and the inference path runs under `torch.inference_mode()` so
no autograd state leaks. It's an ugly mechanism, and the ugliness stays
inside one private function.

**Cost:** `torch.compile` cannot trace this control flow, so the wrapper
refuses it with a clear error instead of miscompiling. Compile the raw backbone
if you need it.

## Honest limits: MobileNetV2 on CIFAR-10

The README publishes a result where earlyon barely helps, on purpose:
MobileNetV2 on CIFAR-10 sends only 8.5% of images out early and still runs
~94% of its FLOPs.

**Why it loses there:** early exit pays off when deep layers are expensive
relative to the heads and the input mix has easy cases to skim off.
MobileNetV2 is already a compressed architecture: its depthwise-separable
blocks make the "skippable" tail cheap, so even a successful early exit saves
little, and the calibrated thresholds respond by exiting rarely to protect
accuracy. On 32×32 CIFAR images upsampled to 224, the effect is amplified.

**Why we publish it anyway:** the signal earlyon reports (average FLOPs used
on real test images, not a theoretical ceiling) is the product. A tool that
only shows its wins teaches users nothing about when to reach for it. The
rule of thumb the numbers support: the bigger and deeper the backbone, the
more earlyon saves (ResNet50 > ResNet18 > MobileNetV2), and compression plus
early exit stack, so wrap the compressed model rather than choosing between
them.
