# Release readiness report — earlyon 0.3 release candidate

Date: 2026-07-22.

## 1. Executive status

**COMPLETE.** Every hardening claim from the handoff was independently
re-verified (and, where under-tested, adversarially tested); one packaging
defect and one evidence-runner defect were found and fixed; the package
builds, twine-checks, installs into a clean environment and passes an
out-of-repo smoke; v1 checkpoint migration is proven against a genuine
prior-version file; a bounded, seeded CUDA evidence run produced real
CIFAR-10 numbers with the negatives preserved. No remote state was touched;
no Jetson/TensorRT claims exist because no such hardware exists here.

## 2. Branches / commits

- Starting point: `claude/earlyon-hardening` @ `db3832e` (8 hardening
  commits on `main` @ `a5f8414`), working tree clean, untouched.
- Work branch: `claude/earlyon-release-readiness` (created from hardening
  HEAD; hardening commits preserved, not rebased/squashed).
- Final HEAD: the last commit listed in §16.

## 3. Environment

Python 3.13.9 · torch 2.10.0+cu128 · torchvision 0.25.0+cu128 · CUDA 12.8 ·
NVIDIA RTX 4050 Laptop GPU (6 GB) · onnx 1.21.0 / onnxruntime 1.25.1 ·
fvcore 0.1.5 · Linux 6.8. **Absent:** tegrastats, TensorRT, Jetson (all
verified absent; nothing was claimed from them).

## 4. Claims independently verified

All of the handoff's headline claims reproduced from scratch:

- 261 tests passing at baseline (now 270), 95% coverage (badge said 96% —
  corrected), strict mypy clean, ruff/black/isort clean for the CI scope.
- **Per-head temperatures (A):** independent per-head fit with convergence
  status; deterministic head mapping; routing applies each head's own T on
  the single-sample, batched (newly pinned) and staged paths; persisted in
  v2 checkpoints; scalar v1 migrates deterministically; invalid values
  raise / fall back safely. New decisive test: a single global temperature
  demonstrably misroutes (accuracy 1.0 → <0.5) where per-head does not.
- **Explicit enablement (B):** disabled exits cannot fire at confidence
  exactly 1.0 or entropy exactly 0.0 (both policies, both paths); all-
  disabled routing reaches the final head; no sentinel thresholds remain;
  migration preserves state.
- **Staged calibration (C):** collect/fit/search/evaluate are separable and
  contracted; simulator pinned bit-exact against the real router; no test
  leakage (split discipline documented and followed by the evidence run);
  search is deterministic (newly pinned); edge cases covered: empty loader
  raises, single-class data works, no-feasible-threshold → disabled,
  NaN-emitting head safely disabled (new), NaN logits rejected in fitting,
  non-converged fits fall back to 1.0 with status.
- **Benchmark fairness (D):** identical preloaded samples/order/warmup/
  sync/boundary proven by spy-module tests; batch-1 validated; model-only
  vs end-to-end labelled; negative speedups preserved (ratio, no clamp);
  full backbone + early-exit + static baseline compared in the evidence run.
- **Compute estimates (E):** `estimated_backbone_flops_fraction` naming,
  `FlopsEstimate` provenance, exit-head exclusion disclosed, reused-module
  detection → warned low-confidence fallback, lazy + cached analysis
  (construction fast). Cache invalidation limitation documented (§17).
- **Checkpoint v2 (F):** format/library version, exit-point identity
  cross-check, routing/enablement/temperatures/loss metadata, deterministic
  v1 migration, future-version rejection, malformed-file rejection,
  `weights_only=True` trust model documented in SECURITY_REVIEW.md. Newly
  verified against a **genuine** prior-version checkpoint written by the
  actual historical `save_wrapper` (committed fixture, ~1.2 MB, provenance
  documented).
- **custom_ee (G):** CPU + CUDA (GPU test passes on this host), kwargs,
  tuple and dict outputs via feature extractors (dict newly pinned),
  missing/reused/out-of-order exits fail loudly; claims match tested scope
  (single-tensor runtime contract stated in the docstring/README).
- **Jetson (H):** restartability, double-start no-op, thread/subprocess/pipe
  cleanup, missing-tool degradation, conservative parsing (None ≠ 0),
  trapezoidal integration with correct units, instantaneous vs average vs
  integrated energy kept distinct — all covered by hardware-free tests. No
  non-Jetson output is presented as Jetson evidence anywhere.
- **Staged deployment (I):** newly proven that later stages are *never
  executed* when an exit fires (counting-module test); staged == eager
  predictions/exit choices across policies, temperatures, enablement;
  unsupported graphs rejected; all-exits ONNX vs conditional execution
  distinction explicit in API name and docs.

## 5. Defects discovered (beyond the handoff)

1. **sdist shipped a partial test tree** (test modules without the fixtures
   package) — broken if anyone ran tests from the sdist.
2. **The v1-migration fixture wasn't actually tracked** (`*.pth` gitignore),
   so the committed migration test would fail on a fresh clone.
3. **Evidence runner OOM'd on 6 GB GPUs** (MobileNetV2 at batch 128 with the
   wrapper still resident).
4. Lint/format debt in `scripts/` and `examples/` (outside CI's checked
   scope): unused import, E402s, ambiguous lambda names, unformatted files.
5. README coverage badge overstated coverage (96% vs measured 95%).

## 6. Fixes made

(1) `MANIFEST.in` prunes tests from the sdist. (2) fixture force-added with
a provenance README. (3) baseline batch 32 + off-GPU staging between phases.
(4) fixed and brought into the format gate. (5) badge corrected. All
conservative; no behavior changes to library code were needed — the audit
found no functional defect in the hardening work itself.

## 7. Tests, coverage, static checks (final)

- `pytest` (full, incl. 6 CUDA tests): **270 passed**, ~13 s.
- Coverage: **95%** (`--cov=earlyon`).
- `mypy earlyon` (strict): clean, 28 files.
- `ruff check earlyon tests scripts examples`: clean.
- `black --check` / `isort --check-only` over the same scope: clean.

## 8. Packaging

`python -m build` → `earlyon-0.2.0-py3-none-any.whl` (73 KB) +
`earlyon-0.2.0.tar.gz`; both `twine check` PASSED; `pip check` clean.
Wheel installed into a fresh virtualenv and smoked **from outside the
repo** (`scripts/smoke_test.py`): import/version, model build, training +
routed inference, deterministic calibration, v2 round-trip, v1 migration,
staged==eager, benchmark smoke — all PASS; CLI entry point and all
subcommand `--help`s work from the installed console script. Caveat: the
clean venv used `--system-site-packages` for the torch stack (a from-scratch
torch download was out of scope); earlyon itself and its metadata were
installed with `--no-deps` from the wheel.

## 9. v1 migration result

PASS — against both hand-built payloads and the genuine historical fixture:
scalar temperature 1.6 broadcast to all four heads, sentinel threshold 1.0
→ `enabled_exits=[True, False, True]`, warning emitted, model routes,
re-save produces a silent v2 round-trip.

## 10. CUDA evidence

Run completed (seed 42, 2+2+2 epochs, 10.4 min compute; full data in
`docs/evidence/cuda_evidence.json`, interpretation in
`docs/evidence/CUDA_EVIDENCE.md`):

- ResNet-18 backbone 94.07% test acc, 1.33 ms p50, 712 ips.
- Early-exit ResNet-18 92.92% (−1.15% vs the 1% budget calibrated on the
  calibration split — generalization gap reported), est. FLOPs fraction
  0.88, **1.10× throughput but worse median latency** (1.43 ms) with better
  tails; 30.5% of images exited early.
- Static MobileNetV2 92.67%, 0.99× — competitive, as the docs predict at
  tiny training budgets.
- Noise-input 2.05× recorded separately as a best-case bound. Peak CUDA
  memory 350 MB during the timed comparison. No result was tuned on test
  data; no negative was suppressed.

## 11. Jetson / TensorRT status

No hardware, no runtime, therefore **no claims**: monitor lifecycle, parser
and energy integration are verified hardware-free; the exact device
procedure is documented (`docs/STAGED_DEPLOYMENT.md` §Jetson, README
Reproducibility). Restartability on a real Jetson and any TensorRT staged
run remain external work.

## 12. Security / privacy

`SECURITY_REVIEW.md` created. Findings: no secrets/keys/tokens; no private
paths or employer identifiers; datasets and build artifacts gitignored; only
committed binary is the documented 1.2 MB untrained migration fixture;
`torch.load(weights_only=True)` + structural validation on every load path;
single subprocess (tegrastats) with fixed argv and cleanup; no eval/exec/
shell=True; wheel/sdist contain no tests, data or weights.

## 13. Public API changes (this branch)

None. All library-code changes on this branch are tests, scripts, packaging
and docs. (API changes from the hardening branch are summarized in
`docs/MIGRATION.md` and `RELEASE_NOTES_DRAFT.md`.)

## 14. Documentation changes

README (architecture diagram, evidence-run results table with negative-
result interpretation, checkpoint-migration + reproducibility sections,
corrected badge), `docs/MIGRATION.md`, `SECURITY_REVIEW.md`,
`PR_BODY_DRAFT.md`, `docs/evidence/CUDA_EVIDENCE.md`, updated
`RELEASE_NOTES_DRAFT.md` and `CHANGELOG.md`, `tests/fixtures/README.md`.

## 15. Files changed (this branch, 17 files)

- tests: `tests/test_release_audit.py` (new, 10 tests),
  `tests/fixtures/v1_cifar_resnet20.pth` + `README.md` (new)
- scripts: `smoke_test.py` (new), `evidence_run.py` (new),
  `run_benchmarks.py` / `re_evaluate.py` / examples (lint/format only)
- packaging: `MANIFEST.in` (new)
- docs: README, CHANGELOG, RELEASE_NOTES_DRAFT, MIGRATION, SECURITY_REVIEW,
  PR_BODY_DRAFT, docs/evidence/* (this report)

## 16. Local commits (`claude/earlyon-hardening..HEAD`)

1. `5d821aa` chore: bring scripts and examples up to the lint/format gate
2. `de8ce1c` test: independent release audit — adversarial verification
3. `1a0d90f` feat: packaging validation — sdist pruning, wheel smoke, v1 fixture, evidence runner
4. `e381db2` docs: security review, migration guide, PR draft, README architecture
5. `5ce0427` docs: coverage badge fix, changelog, release-notes audit section
6. `e38b073` feat: bounded CUDA evidence run — real seeded CIFAR-10 results
7. (final) docs: release readiness report

## 17. Remaining external work

- Jetson: run `earlyon profile` + the staged TensorRT procedure on a real
  device before making any Jetson claim.
- Optional: full-length `scripts/run_benchmarks.py` (hours) to refresh
  `docs/benchmarks.json` `runs` and retire the legacy README table.
- Torchvision-factory stage maps for `earlyon.staged` (protocol + acceptance
  test exist).
- Known limitation (documented): `wrapper.flops_estimate` caches per
  instance and does not invalidate if the backbone is structurally mutated
  after first inference.
- Version bump to 0.3.0 + changelog dating at actual release time.

## 18. Commands for the user (NOT executed)

```bash
# inspect
git log --oneline main..claude/earlyon-release-readiness
git diff main...claude/earlyon-release-readiness

# push + PR
git push -u origin claude/earlyon-release-readiness
gh pr create --base main --head claude/earlyon-release-readiness \
  --title "earlyon 0.3: correctness hardening + release readiness" \
  --body-file PR_BODY_DRAFT.md

# after merge: version bump, tag, publish
#   edit pyproject.toml + earlyon/__init__.py -> 0.3.0; date the CHANGELOG
git tag -a v0.3.0 -m "earlyon 0.3.0"
git push origin v0.3.0
python -m build && python -m twine check dist/* && python -m twine upload dist/*
```
