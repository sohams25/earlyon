# Overnight report — earlyon hardening (branch `claude/earlyon-hardening`)

Date: 2026-07-22. Environment: Linux, Python 3.13.9, torch 2.10.0+cu128,
CUDA available (laptop GPU), no Jetson, no TensorRT.

## 1. Executive summary

earlyon is now a methodologically sound, deployment-oriented toolkit for
training, calibrating, evaluating and benchmarking early-exit PyTorch CV
models. The overnight work fixed the four correctness defects called out in
the brief (shared temperature, sentinel-threshold "disabling", negative-
temperature clamping, unfair benchmark boundaries), rebuilt calibration as a
staged, testable pipeline with a strict data-split discipline, versioned the
checkpoint format with migration, made every compute number an explicitly
labelled estimate, fixed the tegrastats monitor lifecycle and energy
semantics, extended `custom_ee` safely, and added a verified staged-
deployment contract. Public claims were re-based: "production-ready" is
gone, pre-v0.3 benchmark numbers are quarantined as legacy, batch-1
latency-sensitive inference is stated as the primary use case.

## 2. Baseline (before any change)

Commands and results on `main` (a5f8414):

- `pytest -m "not gpu"` → **177 passed, 5 deselected, ~14 s**
- `mypy earlyon` (strict) → clean
- `ruff check earlyon tests` → clean
- `black --check` / `isort --check-only` on `earlyon tests` → clean
  (`scripts/` was unformatted; CI does not check it — scripts were
  reformatted as part of their rewrite)
- No hanging tests; ViT tests were the slowest (~1 s each) because FLOPs
  analysis ran at construction.

## 3. Major architecture & methodology changes

1. **Per-head temperatures** — `EarlyExitConfig.temperatures: dict[head, T]`
   covering every exit and the final classifier; routing applies each head's
   own T. `fit_head_temperatures` fits per head from one cached-logits pass;
   each fit carries convergence/fallback status (fallback = safe 1.0, never
   a sharp value).
2. **Explicit enablement** — `enabled_exits: list[bool]`; disabled exits
   never fire (confidence exactly 1.0 / entropy exactly 0.0 included) and
   their heads aren't evaluated at inference. Sentinels retired; v1
   checkpoints migrate.
3. **Centralized validation** — `EarlyExitConfig.validate()` (all
   invariants; actionable errors) re-invoked at wrapper construction and
   checkpoint load. Invalid negative temperature now raises instead of the
   old `max(t, 1e-6)` clamp.
4. **Staged calibration pipeline** — collect → fit → search → evaluate as
   separate tested functions; the vectorized simulator is pinned exactly
   against the real router. `CalibrationResult` reports enablement, exit
   distribution, sample count, delta, objective, honest method name, schema
   version. Never-firing exits stay disabled. Empty loaders raise.
5. **Fair benchmark runner** — `benchmark_models`: identical preloaded
   samples, warmup, boundary and sync for every compared model; model-only
   vs end-to-end labelled; accuracy alongside speed for labelled loaders;
   batch-1 validated. CLI + scripts use it; legacy helpers delegate to it.
6. **Honest compute accounting** — `estimated_backbone_flops_fraction`
   (alias `computation_used` kept) from a lazy, cached `FlopsEstimate`
   carrying method/reliability/notes; reused-module detection degrades to a
   warned low-confidence uniform fallback; exit-head cost explicitly
   excluded.
7. **Checkpoint v2** — versioned schema with exit-point cross-check,
   deterministic v1 migration (broadcast temperature; sentinel → disabled),
   newer-format rejection, `factory=` loading for custom models.
8. **Jetson** — restartable monitor (stop event cleared on start — the
   restart bug), `None` for missing telemetry, `integrate_energy` /
   `EnergySummary` separating instantaneous power, window-average power and
   trapezoid-integrated energy; per-inference energy only from a real
   integral.
9. **custom_ee** — example args/kwargs, backbone-device inference, per-layer
   feature extractors (dry run + routing hook), exactly-once and in-order
   exit-layer validation, heads placed on the backbone's device.
10. **Staged deployment** — `earlyon.staged` protocol + reference splitter
    for Sequential backbones, self-verified against the eager wrapper;
    `export_all_exits_to_onnx` alias names the static ONNX export honestly.

## 4. Bugs reproduced and corrected

| Defect (verified against the old code) | Fix |
|---|---|
| One temperature fit from final-classifier logits, reused for every exit (docs claimed per-head) | per-head fit + routing (P0-A) |
| Threshold `1.0` "disabled" exits could fire at saturated confidence (`>=`); entropy `0.0` at entropy 0 | explicit `enabled_exits` (P0-B) |
| Negative/zero temperature clamped to `1e-6` → artificially sharp softmax | validate() raises; runtime guard falls back to 1.0 |
| `computation_used` read as measured FLOPs; module-order assumption unchecked | renamed estimate + reuse detection + provenance |
| Wrapper and backbone benchmarked on different inputs/boundaries | `benchmark_models` fairness contract |
| `TegrastatsMonitor` unrestartable (stale stop event); pipe not closed | lifecycle fix + tests |
| Missing tegrastats fields reported as 0.0 (fabricated) | `None` + energy integration semantics |
| ViT FLOPs analysis at construction (slow) | lazy + cached |
| `custom_ee` blindly probed CUDA models with CPU tensors; accepted reused/out-of-order exit layers | device inference + dry-run validation |
| Calibration could enable an exit at a threshold that never fires | never-firing exits stay disabled |
| Empty calibration loader silently returned 0.0/0.0 | raises ValueError |
| LBFGS `exp(log_t)` could overflow to inf | guarded, falls back with status |

## 5. Files changed by subsystem

- **core**: `types.py` (config redesign, validation, result renames),
  `wrappers.py` (per-head T, enablement, adapters, lazy FLOPs, head/config
  cross-checks), `thresholds.py` (staged pipeline rewrite), `temperature.py`
  (fit status), `flops.py` (FlopsEstimate, reuse detection, device param)
- **benchmarking**: `throughput.py` (rewritten around `benchmark_models`),
  `jetson_profiler.py` (rewritten), `__init__.py`
- **serialization**: `utils.py` (checkpoint v2 + migration + factory)
- **models**: `custom.py` (rewritten)
- **new**: `earlyon/staged.py`, `docs/STAGED_DEPLOYMENT.md`,
  `docs/CALIBRATION_AND_BENCHMARK_CONTRACT.md`, `tests/test_staged.py`,
  `RELEASE_NOTES_DRAFT.md`, this report
- **cli/scripts/examples/docs**: `cli.py` (benchmark/profile commands),
  `scripts/run_benchmarks.py`, `scripts/re_evaluate.py`,
  `docs/benchmarks.json` (legacy quarantine), `examples/01`, `examples/02`,
  `README.md`, `docs/DESIGN_DECISIONS.md`, `CHANGELOG.md`, `pyproject.toml`
- **tests**: substantial additions across `test_types`, `test_wrappers`,
  `test_thresholds`, `test_temperature`, `test_utils`, `test_benchmarks`,
  `test_flops`, `test_jetson_profiler`, `test_custom_ee`, `test_staged`

## 6. Public API / config / checkpoint migration notes

- `EarlyExitConfig`: new `temperatures` (dict) and `enabled_exits` fields;
  `temperature` is constructor-only (broadcast) — mutate `temperatures`.
  Constructing with both `temperature != 1.0` and `temperatures` raises.
- `InferenceResult`/`BatchedInferenceResult`/`BenchmarkResult`/`JetsonRun`:
  FLOPs field renamed; `computation_used` remains a read-only alias.
  Constructing these with `computation_used=` no longer works (library-
  internal construction only).
- `CalibrationResult`: superset of old fields; `fitted_temperature` is now
  the final head's fit (deprecated alias).
- Checkpoints: v2 written; v1 read with deterministic migration + warnings;
  `load_wrapper(..., factory=...)` for custom models; newer formats rejected.
- `benchmark_wrapper` / `benchmark_backbone` / `benchmark_wrapper_on_loader`
  keep signatures (now thin wrappers over `benchmark_models`).

## 7. Tests and final gate results

- Baseline 177 tests → **261 tests**: `pytest` (full, incl. 6 GPU tests on
  the CUDA host) → **261 passed** (~16 s); `pytest -m "not gpu"` → 255
  passed.
- Coverage (CPU subset): **95%** (`--cov=earlyon`).
- `mypy earlyon` (strict): clean, 28 files.
- `ruff check earlyon tests`: clean. `black --check` / `isort --check-only`:
  clean (scripts now formatted too).
- The former ViT slowness: construction no longer runs FLOPs analysis; the
  ViT test module runs in ~2 s dominated by torchvision model build.

## 8. Benchmarks actually run

None published. All numeric records in `docs/benchmarks.json` are the
pre-existing v0.2 measurements, now explicitly quarantined under
`legacy_v0_2` with a note that they are not comparable to the v0.3 fair
runner (the old runner fed the wrapper and backbone different inputs). The
new runner is fully implemented and tested with tiny deterministic models;
regenerating real numbers is one command (`python scripts/run_benchmarks.py`)
on the target machine and was deliberately not run overnight to avoid mixing
a multi-hour training job into a correctness review.

## 9. Hardware checks not run, and the later procedure

No Jetson, no TensorRT, no new CUDA benchmark numbers. Procedure for later:

1. On the Jetson (MAXN, fans steady): `pip install -e .`, verify
   `tegrastats` on PATH.
2. Train/calibrate on a workstation, `save_wrapper`, copy the `.pth`.
3. `earlyon profile --model calibrated.pth --runs 500 --warmup 50` — the
   energy block reports integrated mJ and mJ/inference from real samples.
4. `python scripts/re_evaluate.py` for the fair wrapper-vs-backbone table.
5. TensorRT: per-stage export + `trtexec` loop per
   `docs/STAGED_DEPLOYMENT.md` (procedure documented, nothing fabricated).

## 10. Remaining limitations / deferred work

- Staged splitting covers Sequential backbones only; torchvision factories
  need per-architecture stage maps (protocol + acceptance test exist).
- Masked per-sample batched routing is still roadmap.
- Per-stage ONNX export helper is documented but not implemented as a
  one-call function (each `Stage` traces cleanly with `torch.onnx.export`).
- README benchmark table still shows legacy v0.2 numbers (labelled);
  regenerate before release.
- `evaluate`/`AccuracyReport` still uses the `avg_computation_used` name
  (documented as an estimate); renaming it was skipped to limit churn.

## 11. Suggested next version & release checklist

**0.3.0** — see `RELEASE_NOTES_DRAFT.md` (includes the checklist).

## 12. Local commits (oldest first)

1. `3e69bbc` feat: per-head temperatures, explicit exit enablement, staged
   calibration, checkpoint v2
2. `5f03a98` feat: fair benchmark runner with identical boundaries and inputs
3. `5724afe` feat: honest compute accounting — lazy FLOPs estimate with provenance
4. `c732544` fix: restartable tegrastats monitor, honest power/energy semantics
5. `7b84818` feat: safer, more capable custom_ee
6. `43e82b1` feat: staged deployment contract with verified reference splitter
7. `a750586` fix: cast ModuleList access in staged runtime for strict mypy
8. (docs commit) docs: repositioning, contracts, changelog, release draft,
   overnight report

## 13. Five-minute interview explanation

**Early-exit architecture.** A CNN spends the same compute on every image,
but most images are easy. earlyon attaches small classifier heads (~10-100k
params) at intermediate layers of an unmodified backbone via forward hooks.
At inference, each enabled head computes a softmax; if its calibrated
criterion passes (max-probability above a threshold, or entropy below one),
a sentinel exception unwinds the forward and that head's logits are the
answer — later layers never run. Training is two-stage: train the backbone
normally, freeze it (parameters *and* BatchNorm stats), then train the heads
for a few epochs with a weighted multi-exit cross-entropy.

**Calibration methodology.** Thresholding raw softmax is meaningless because
each head is differently over-confident, so we first fit one temperature per
head (Guo et al. 2017) on a held-out split — NLL minimization that rescales
confidence without changing predictions. Then a greedy per-exit grid search
on a separate calibration split picks the most aggressive thresholds that
keep routed accuracy within the user's budget (or meet a compute budget with
least accuracy loss). The search runs the network once — logits are cached
per head and every grid point is simulated vectorially, pinned bit-exact
against the real router by a test. Exits that never fire stay explicitly
disabled. Test labels are never visible to the search; reported accuracy
comes from a separate evaluation pass.

**Why FLOP savings may not become latency savings.** The FLOPs number is a
static estimate of the backbone fraction executed — it excludes the exit
heads and, critically, the routing overhead: each enabled exit computes a
softmax and calls `.item()`, which synchronizes the GPU pipeline. On a fast
GPU with a small backbone, one host sync can cost more than the skipped
layers; kernels are also more efficient in bulk. So a model can "save" 20%
of estimated FLOPs and be *slower* in wall-clock — our own legacy ResNet50
row shows exactly that (0.94× on real inputs).

**How the project measures this honestly.** One benchmark core feeds every
compared model — backbone, wrapper, and ideally a smaller static model at
matched accuracy — the exact same preloaded sample sequence, same warmup,
same synchronization, with the measurement boundary labelled (model-only vs
end-to-end). Estimated FLOPs and wall-clock are reported side by side and
never conflated; noise-input runs are labelled best-case; results that
predate this methodology are quarantined as legacy; and on Jetson, energy is
only reported as the time-integral of real power samples over the timed
window — never extrapolated from a single instantaneous reading.
