# Calibration & benchmark contract

The rules earlyon holds itself to when it fits a routing policy and when it
publishes a number. Everything here is enforced by code and pinned by tests.

## Data splits

| Split | Used for | Never used for |
|---|---|---|
| train | backbone + exit-head training | calibration, evaluation |
| temperature (held-out; optional) | per-head temperature fitting | threshold search |
| calibration / validation | threshold search, enablement, calibration-split metrics | reported accuracy claims |
| test | final reported accuracy + exit distribution (`earlyon.benchmarking.evaluate`) | anything the search touches |

`calibrate_thresholds(fit_temperature=True)` without a separate
`temperature_loader` reuses the calibration split and **warns** — the
temperature fit leaks into the threshold accuracy estimate. All accuracies in
a `CalibrationResult` are calibration-split numbers, labelled as such.

## Calibration pipeline (four separable stages)

1. **Collect** — `collect_head_logits` runs the loader once through the
   all-exits forward, caching raw logits for every head (exits + final).
   Empty loaders raise immediately.
2. **Fit temperatures** — `fit_head_temperatures`: one temperature per head
   (each head is a differently miscalibrated classifier). Fits carry a
   convergence/fallback status; a diverged fit falls back to 1.0 (no
   calibration) with a warning — never to an artificially sharp value.
3. **Search** — greedy coordinate descent over a fixed grid, simulated
   against the cache. It is named `greedy-coordinate-grid` in results because
   that is what it is; no global-optimality claim is made. A threshold that
   fires no calibration sample leaves its exit **disabled**
   (`enabled_exits[i] = False`) — an explicit boolean, not a sentinel, so a
   float-saturated softmax (confidence exactly 1.0, entropy exactly 0.0) can
   never fire a disabled exit.
4. **Evaluate** — the selected policy is scored on the calibration split
   (accuracy, estimated compute, exit distribution) into `CalibrationResult`.
   The simulator is pinned bit-for-bit against the real routing path by
   `test_eval_cache_matches_real_router_exactly`.

## Compute numbers are estimates

`estimated_backbone_flops_fraction` (formerly `computation_used`, kept as a
read alias) comes from a static fvcore analysis of the backbone:

- exit-head FLOPs and all routing overhead are **excluded**;
- the per-layer attribution assumes each leaf module runs exactly once, in
  registration order (true for the shipped torchvision backbones). A reused
  module is detected and degrades the estimate to a warned, low-confidence
  uniform fallback (`FlopsEstimate.reliable = False`);
- the analysis is lazy (first inference), cached, and its provenance is on
  `wrapper.flops_estimate`.

A FLOPs saving is **not** a latency claim. The eager router evaluates an
exit head and synchronises the host (`.item()`) at every enabled exit; on a
GPU that synchronisation can cost more than the skipped layers save,
especially on small backbones. That is why benchmark results and the README
report wall-clock and estimated-FLOPs numbers separately.

## Benchmark fairness rules (`benchmark_models`)

1. Every compared model sees the **exact same preloaded sample sequence** —
   same order, same tensors — with the same warmup count.
2. Boundaries are identical and labelled: `model-only` (input staged on the
   device before timing) or `end-to-end` (H2D copy inside the timed region).
   Loader/preprocessing time is never silently included.
3. Same `eval()` + `torch.inference_mode()` + per-iteration CUDA
   synchronisation for every model.
4. Batch 1 is the primary supported mode (the latency-sensitive edge case
   earlyon targets) and is validated.
5. Reported: p50/p95/p99 latency, throughput, run count, device, dtype,
   input shape, warmup, sync policy, exit distribution, input source
   (`loader` vs `random-noise`), and accuracy when the loader is labelled.
6. Random-noise input is labelled a best-case bound: trained heads can fire
   spuriously on noise.
7. A speedup is only quoted between results produced by the same
   `benchmark_models` call. Pre-v0.3 records in `docs/benchmarks.json` live
   under `legacy_v0_2` and are marked not comparable (the old runner fed the
   wrapper and backbone different inputs).
8. An early-exit model should also be compared against a **smaller static
   model** at matched accuracy — if a ResNet18 matches your routed ResNet50's
   accuracy and latency, the honest deployment is the ResNet18. Pass it as a
   third entry to `benchmark_models`.

## Serialization

Checkpoints are `format_version: 2`: routing policy, both threshold lists,
`enabled_exits`, per-head `temperatures`, loss weights, exit-point metadata
(cross-checked against the rebuilt factory), library version. Unversioned v1
files migrate deterministically (scalar temperature broadcast; sentinel
thresholds → explicit disabled) with warnings. `custom_ee` models reload only
via a user-supplied `factory=` — an arbitrary backbone is never silently
"reconstructed".
