# Public release report — earlyon v0.3.0

Date: 2026-07-22.

## Status

**Released on GitHub; PyPI publication blocked on one external step.**
The PR is merged, `v0.3.0` is tagged and has a GitHub Release, the published
docs preserve every negative result and hardware limitation, and a fresh
environment installs and smokes cleanly from the public tag. The
tag-triggered PyPI publish failed because **no PyPI trusted publisher (or
even project) exists for `earlyon`** — an account-level, 2FA-gated setup
that cannot be done from this environment (see Blocker).

## Facts

| | |
|---|---|
| Repository | https://github.com/sohams25/earlyon (public, owner `sohams25`) |
| PR | https://github.com/sohams25/earlyon/pull/5 — MERGED |
| Merge commit | `19e8072ad4d63386ebf87a6773b78514c559f851` (merge commit; 16-commit history preserved, nothing squashed or force-pushed) |
| Tag | `v0.3.0` → annotated `a18ffbf`, points at the merge commit |
| GitHub Release | https://github.com/sohams25/earlyon/releases/tag/v0.3.0 (from the finalized release notes) |
| PyPI | **Not published** — workflow run 29919523337: `invalid-publisher` (no trusted publisher configured; the v0.1.0 release job failed the same way in May 2026, so `earlyon` has never been on PyPI) |

## Public install verification (fresh venv, outside the repo)

`pip install git+https://github.com/sohams25/earlyon@v0.3.0` (public tag;
the PyPI path is unavailable per the blocker):

- import + `__version__ == "0.3.0"` — PASS
- built-in model creation, training forward, routed inference — PASS
- deterministic calibration, **v2 checkpoint round-trip, v1 migration** — PASS
- staged reference execution == eager — PASS
- benchmark smoke — PASS
- installed CLI (`earlyon --version`, wrap/calibrate/benchmark/export help) — PASS

No CIFAR retraining was performed during verification.

## CI / tests / coverage

- PR CI: `test (3.10)`, `test (3.11)`, `test (3.12)` all PASS (lint + black
  + isort + mypy strict + pytest with coverage inside each job).
- Local at the merged commit: 270 tests pass (264 CPU + 6 CUDA on this
  host), coverage 95%, mypy strict clean, ruff/black/isort clean.
- Post-merge: build + `twine check` PASSED on artifacts from `19e8072`;
  wheel smoke (incl. v1 migration) PASS.

## Release-only changes

One commit: `d13d240` `chore(release): prepare v0.3.0` — version 0.3.0 in
`pyproject.toml` / `earlyon/__init__.py` / README badge + citation,
CHANGELOG dated, release notes finalized (limitations, migration,
bounded-evidence wording), PR body header finalized. No release-only code
fixes were needed; CI passed on the first run.

## Honesty checks on the published pages (verified post-publish)

The rendered v0.3.0 README/release notes state: the 1.10× throughput figure
is one seeded bounded run; the early-exit **median latency is worse** than
the full backbone (tails better); the static MobileNetV2 baseline is
competitive; noise-input 2.05× is labelled a best-case bound only; compute
fractions are estimates, not latency/energy; all-exits ONNX is not
conditional execution; staged execution has a narrow supported contract;
the Jetson procedure is documented but **unmeasured**; TensorRT is not
claimed; no universal speedup claim exists.

## Blocker (one, external)

**PyPI trusted publisher setup requires an interactive PyPI account login
(password + 2FA), which is unavailable here.** Exact remediation, once, by
the account owner:

1. Log in to pypi.org → *Publishing* → *Add a pending publisher* with
   exactly: project `earlyon`, owner `sohams25`, repository `earlyon`,
   workflow `release.yml`, environment `pypi` (these match the OIDC claims
   the failed run printed).
2. Re-run the existing failed job — no re-tagging needed:
   `gh run rerun 29919523337 -R sohams25/earlyon`
   (artifacts were already built from the tagged commit; only the publish
   job re-executes).
3. Verify: https://pypi.org/project/earlyon/ shows 0.3.0, then
   `python -m venv /tmp/e && /tmp/e/bin/pip install earlyon==0.3.0` and run
   `scripts/smoke_test.py` from outside the repo.

## Remaining unmeasured hardware evidence

Jetson (tegrastats energy, staged TensorRT) — procedure documented in
`docs/STAGED_DEPLOYMENT.md`; no numbers exist and none are claimed.

## Maintenance recommendation

Freeze broad feature work. Focus on: user reports against 0.3.0,
compatibility fixes (new torch/torchvision releases), completing the PyPI
publisher setup above, and producing real Jetson evidence before any
hardware claim. The v0.3 legacy benchmark table in the README should be
retired only after a full-length `scripts/run_benchmarks.py` run on target
hardware.
