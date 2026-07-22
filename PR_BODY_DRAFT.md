# Release v0.3.0: defensible calibration, fair benchmarks, and staged inference

> Base: `main`. Head: `claude/earlyon-release-readiness`
> (includes the `claude/earlyon-hardening` work).

## What this PR is

A correctness and methodology release. No new backbones, no new headline
claims — the calibration math, the measurement methodology, the
serialization contract and the public claims are made defensible, and every
fix is pinned by tests (261 → 265+ tests, 95% coverage, strict mypy).

## The four core defects fixed

1. **One temperature for every head.** Temperature scaling was fit once from
   the final classifier's logits and reused at every exit, though each exit
   head is a differently miscalibrated classifier. Now: per-head fit with
   convergence status, per-head application at routing time. A test proves a
   single global temperature misroutes where per-head does not.
2. **"Disabled" exits could fire.** Disabling used threshold sentinels
   (confidence 1.0 / entropy 0.0) while routing compares with `>=`/`<=`, so
   a float32-saturated softmax fired a "disabled" exit. Now: explicit
   `enabled_exits` booleans; sentinel-carrying v1 checkpoints migrate.
3. **Invalid temperatures were clamped sharp.** A negative fitted/user
   temperature was clamped to `1e-6`, producing an artificially razor-sharp
   softmax. Now: centralized `EarlyExitConfig.validate()` raises; runtime
   guard falls back to the no-op 1.0.
4. **Benchmarks weren't comparable.** The wrapper and backbone were measured
   on different inputs and boundaries. Now: `benchmark_models` feeds every
   compared model the identical preloaded sample sequence with identical
   warmup/boundary/sync; old records quarantined as `legacy_v0_2`.

## Also in this PR

- Staged calibration pipeline + rich `CalibrationResult` (enablement, exit
  distribution, sample count, honest `greedy-coordinate-grid` method name).
- Checkpoint `format_version: 2` + deterministic v1 migration, tested
  against a genuine v0.2-written fixture; `factory=` loading for custom
  models.
- `estimated_backbone_flops_fraction` (alias kept): lazy `FlopsEstimate`
  with provenance, reused-module detection, exit-head cost disclosed.
- Restartable tegrastats monitor; instantaneous power vs window-average vs
  integrated energy kept distinct; missing telemetry is `None`.
- `custom_ee`: example args/kwargs, device inference, feature extractors,
  exactly-once/in-order exit validation.
- Staged deployment contract + reference splitter for Sequential backbones,
  proven equivalent to eager routing and proven to skip later stages.
- Packaging: wheel/sdist build + twine-clean; clean-env wheel smoke script;
  sdist no longer ships a partial test tree.
- Bounded, seeded CUDA evidence run on CIFAR-10 (ResNet-18 EE vs backbone vs
  static MobileNetV2) with disjoint train/temperature/calibration/test
  splits — results in `docs/evidence/`.
- Docs: calibration/benchmark contract, migration guide, staged deployment
  guide, security review, honest README repositioning
  ("deployment-oriented", legacy labels, batch-1 scope).

## Compatibility

Old checkpoints load (migrated, warned). `computation_used`,
`fitted_temperature` and the single-model benchmark helpers remain as
compatible aliases/wrappers. Breaking edges are listed in
`docs/MIGRATION.md`.

## Test plan

- [x] `pytest` full suite (CPU + CUDA host): all green
- [x] `mypy earlyon` strict, `ruff`, `black --check`, `isort --check-only`
- [x] wheel + sdist build, `twine check`, clean-venv install + out-of-repo
      smoke (`scripts/smoke_test.py`)
- [x] genuine v1 checkpoint migration fixture
- [x] bounded CUDA evidence run (`scripts/evidence_run.py`)
- [ ] optional before tagging: full-length `scripts/run_benchmarks.py` on
      target hardware; Jetson procedure from `docs/STAGED_DEPLOYMENT.md`
