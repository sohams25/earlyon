# earlyon 0.3.0 — draft release notes (NOT published, NOT tagged)

Suggested version: **0.3.0** (routing/calibration semantics changed;
checkpoints migrate automatically, but calibrated behavior can differ —
disabled exits are now strictly disabled).

## Headline

earlyon 0.3 is a correctness release. It makes the calibration methodology
defensible (per-head temperatures, explicit exit enablement, split
discipline), the measurements fair (one benchmark core, identical samples
and boundaries for every compared model), and the claims honest (estimated
FLOPs labelled as estimates, legacy numbers quarantined, "production-ready"
retired in favor of "deployment-oriented").

## Breaking / behavioral changes (all with migration paths)

- One temperature **per head** (`config.temperatures`), fitted per head when
  `fit_temperature=True`. The scalar `temperature` field is a constructor
  convenience only; mutate `temperatures` after construction.
- Disabled exits are explicit booleans (`config.enabled_exits`). Threshold
  sentinels are gone; a saturated softmax can no longer fire a "disabled"
  exit. v1 checkpoints migrate their sentinels with a warning.
- Calibration leaves never-firing exits **disabled** instead of enabled at
  an idle threshold; `CalibrationResult` gained enablement, exit
  distribution, sample count, objective and method fields.
- `computation_used` → `estimated_backbone_flops_fraction` (read alias
  kept). It is an estimate excluding exit heads and routing overhead.
- `BenchmarkResult.avg_computation_used` → `avg_estimated_flops_fraction`
  (read alias kept); results now carry boundary/sync/input-source metadata.
- Checkpoints are `format_version: 2`; v1 loads migrate deterministically.
  `custom_ee` models reload via `load_wrapper(path, factory=...)`.
- Invalid config now raises from centralized `EarlyExitConfig.validate()`
  (previously e.g. a negative temperature was clamped to `1e-6`).

## New

- `benchmark_models`: fair multi-model comparison (backbone / wrapper /
  static baseline / quantised variant) on identical samples, with accuracy
  reported for labelled loaders and explicit model-only vs end-to-end
  boundaries.
- `earlyon.staged`: staged-deployment protocol + verified reference splitter
  for Sequential backbones; staged execution matches eager routing and
  genuinely skips later stages. `docs/STAGED_DEPLOYMENT.md` documents the
  contract and the Jetson/TensorRT procedure.
- Jetson: restartable `TegrastatsMonitor`, `None` for missing telemetry,
  `profile_with_energy` with real integrated energy (`EnergySummary`).
- `custom_ee`: example args/kwargs, device inference, per-layer feature
  extractors, and dry-run validation (exactly-once, in-order exits).
- Lazy `FlopsEstimate` with provenance; reused-module detection.
- `docs/CALIBRATION_AND_BENCHMARK_CONTRACT.md`.

## Release-readiness verification (post-hardening audit)

The hardening claims were re-verified independently on
`claude/earlyon-release-readiness`:

- adversarial tests added, including the decisive counterfactual that a
  single global temperature misroutes where per-head temperatures do not,
  batched-path per-head temperatures, NaN-emitting heads, calibration
  determinism, and structural proof that the staged runtime skips later
  stages;
- v1→v2 checkpoint migration verified against a **genuine** v0.2-written
  fixture (`tests/fixtures/v1_cifar_resnet20.pth`), not a hand-built dict;
- wheel + sdist build `twine`-clean; wheel installed into a clean venv and
  smoked from outside the repo (`scripts/smoke_test.py`); sdist no longer
  ships a partial test tree;
- a bounded, seeded CUDA evidence run (`scripts/evidence_run.py`) on
  CIFAR-10 with disjoint train/temperature/calibration/test splits and a
  static MobileNetV2 baseline — results in `docs/evidence/`;
- security review in `SECURITY_REVIEW.md`; migration guide in
  `docs/MIGRATION.md`.

## Release checklist (when actually releasing)

- [ ] bump `__version__` and `pyproject.toml` to 0.3.0; move `unreleased` in
      CHANGELOG under 0.3.0 with a date
- [ ] regenerate `docs/benchmarks.json` `runs` with
      `scripts/run_benchmarks.py` on real hardware; update the README table
      from those records and drop the legacy labels for the new numbers
- [ ] run the Jetson procedure in `docs/STAGED_DEPLOYMENT.md` §Jetson if a
      device is available (optional for release, required before any Jetson
      claim)
- [ ] full gate: `pytest -m "not gpu"`, `pytest -m gpu` (CUDA host),
      `mypy earlyon`, `ruff check earlyon tests`, `black --check`,
      `isort --check-only`
- [ ] `python -m build` + twine check; follow `docs/PYPI_RELEASE_CHECKLIST.md`
