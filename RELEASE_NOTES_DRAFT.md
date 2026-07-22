# earlyon v0.3.0 — release notes

(This file is the canonical text for the v0.3.0 GitHub Release.)

earlyon 0.3.0 is a correctness release for the early-exit toolkit. It makes
the calibration methodology defensible (per-head temperatures, explicit exit
enablement, strict data-split discipline), the measurements fair (one
benchmark core, identical samples and boundaries for every compared model),
and the claims honest (estimated FLOPs labelled as estimates, legacy numbers
quarantined, "production-ready" retired in favor of "deployment-oriented").

## Install

```bash
pip install earlyon==0.3.0
```

Python 3.10+, torch 2.0+. Existing checkpoints load with automatic
migration — see **Migration** below.

## Methodological corrections

- **One temperature per head.** Temperature scaling used to be fit once from
  the final classifier's logits and reused at every exit, though each exit
  head is a differently miscalibrated classifier. 0.3 fits and applies a
  temperature per head (`config.temperatures`), with per-fit convergence
  status; a regression test proves a single global temperature misroutes
  where per-head fitting does not.
- **Disabled exits cannot fire.** Disabling via threshold sentinels
  (confidence `1.0` / entropy `0.0`) was numerically unsound: a
  float32-saturated softmax met the criterion. Enablement is now an explicit
  boolean per exit (`config.enabled_exits`), honored by the confidence and
  entropy policies on the single-sample, batched and staged paths.
- **Staged calibration pipeline.** Collect logits once → fit per-head
  temperatures → greedy threshold grid search on the cache → evaluate; each
  stage separately testable, the simulator pinned bit-exact against the real
  router, threshold search never sees test labels, never-firing exits stay
  disabled, empty loaders fail fast. The method is named what it is
  (`greedy-coordinate-grid`) — no global-optimality claim.
- **Centralized validation.** `EarlyExitConfig.validate()` raises actionable
  errors for invalid config (a negative temperature used to be silently
  clamped to `1e-6`, producing an artificially sharp softmax).

## Fair benchmarking

`benchmark_models` measures every compared model — full backbone, early-exit
wrapper, a static smaller baseline, a quantised variant — on the **exact
same preloaded sample sequence** with identical warmup, boundary
(model-only vs end-to-end, labelled) and synchronization, reporting
p50/p95/p99, throughput, exit distribution and accuracy for labelled
loaders. Pre-0.3 benchmark records are quarantined as `legacy_v0_2` in
`docs/benchmarks.json`: they fed compared models different inputs and are
not comparable.

## Honest compute terminology

`computation_used` is renamed `estimated_backbone_flops_fraction` (read
alias kept). It is a static estimate — it excludes exit-head cost and all
routing overhead, and it is neither a latency nor an energy measurement.
The estimator (`FlopsEstimate`) reports its method and reliability, detects
reused modules (degrading to a warned low-confidence fallback), and runs
lazily so large-model construction is fast.

## Staged execution

`earlyon.staged` defines a staged-deployment contract (ordered stages
emitting continuation features + exit logits, routing applied between
stages) with a verified reference splitter for `nn.Sequential` backbones.
Staged execution provably matches eager routing and *structurally* skips
later stages — unlike the all-exits ONNX export
(`export_all_exits_to_onnx`), which always computes every exit and is a
portability artifact, not conditional execution. The supported contract is
deliberately narrow; unsupported graphs are rejected. See
`docs/STAGED_DEPLOYMENT.md`.

## Checkpoints and migration

Checkpoints now carry `format_version: 2` (exit-point identity, routing
policy, thresholds, enablement, per-head temperatures, library version) and
are cross-checked against the rebuilt factory on load. Files written by
earlyon ≤ 0.2 migrate automatically and deterministically — scalar
temperature broadcast per head, legacy sentinels → explicit disables — with
a warning; the migration is pinned against a genuine v0.2-written fixture.
`custom_ee` models reload via `load_wrapper(path, factory=...)`.
Details and code-level changes: `docs/MIGRATION.md`.

## Also fixed

- `custom_ee`: example args/kwargs for inspection, device inference (no more
  CPU probes of CUDA models), per-layer feature extractors for
  tuple/dict/odd-rank outputs, and dry-run validation that every exit layer
  runs exactly once in the listed order.
- Jetson monitoring: `TegrastatsMonitor` is restartable (a restarted monitor
  previously collected nothing), missing telemetry parses to `None` instead
  of invented zeros, and `profile_with_energy` distinguishes instantaneous
  power, window-average power and trapezoid-integrated energy.

## Real CUDA evidence (one seeded, bounded run — not a universal claim)

A deliberately small validation run (CIFAR-10 at 224px, ResNet-18 EE
fine-tuned 2+2 epochs from ImageNet weights, seed 42, disjoint
train/temperature/calibration/test splits, RTX 4050 Laptop GPU; full data in
`docs/evidence/`):

- early-exit ResNet-18: 92.92% test accuracy vs 94.07% full backbone;
  **1.10× throughput** with **worse median latency** (1.43 vs 1.33 ms) and
  better tail latency (p95/p99) — routing overhead is real on the ~70% of
  images that ran the full network;
- a static MobileNetV2 baseline trained the same way was competitive
  (92.67%, 0.99×) — smaller static models remain an essential comparison;
- a noise-input 2.05× figure exists only as a labelled best-case bound, not
  a practical speed claim.

Early exit is **not always faster**; the fair runner exists so you can see
when it isn't.

## Limitations

- Dynamic routing is primarily eager PyTorch execution (hooks + host sync);
  `torch.compile` is refused with a clear error.
- All-exits ONNX export computes every exit — no conditional execution.
- Staged reference execution supports Sequential backbones only.
- The Jetson/tegrastats procedure is documented but **unmeasured** — no
  Jetson numbers exist; TensorRT deployment is not claimed.
- Estimated FLOPs fractions are estimates, not measured latency or energy.
- `forward(mode="inference")` is batch-1; batched routing is conservative.

## Verification behind this release

270 tests (incl. CUDA suite on a CUDA host), 95% coverage, mypy strict,
ruff/black/isort clean; wheel + sdist twine-clean and smoked from a clean
environment (`scripts/smoke_test.py`); security review in
`SECURITY_REVIEW.md`; full audit trail in `RELEASE_READINESS_REPORT.md`.
