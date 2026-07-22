# PyPI Release Checklist

Releases are driven by git tags: pushing a `v*` tag runs
`.github/workflows/release.yml`, which builds and publishes to PyPI.
**Publishing is irreversible** (you can yank, but never reuse a version
number), so walk this list top to bottom before tagging. Tagging is a
manual, human decision — never automated.

## 1. Pre-flight

- [ ] Working tree clean and on `main`, up to date with `origin/main`:
      `git status && git pull --ff-only`
- [ ] CI green on `main` (all jobs, not just tests):
      check the Actions tab or `gh run list --branch main --limit 1`
- [ ] Full test suite passes locally, including slow tests:
      `pytest tests/ -v --cov=earlyon -m "not gpu"`
- [ ] GPU tests pass if hardware is available: `pytest tests/ -v -m gpu`
- [ ] Type check clean: `mypy earlyon/ --strict`
- [ ] Lint/format clean: `black --check . && isort --check .`

## 2. Version and metadata

- [ ] Bump `version` in `pyproject.toml` (semver: breaking → major,
      feature → minor, fix → patch)
- [ ] `earlyon/__init__.py` version matches `pyproject.toml`
- [ ] `CHANGELOG.md` has a dated section for this version covering every
      user-visible change since the last tag
      (`git log $(git describe --tags --abbrev=0)..HEAD --oneline` to audit)
- [ ] README still accurate: `pip install earlyon` present, version badge
      updated, benchmark numbers match `docs/benchmarks.json`
- [ ] No stray files that would ship: `git status --ignored` shows nothing
      unexpected outside `.gitignore`

## 3. Build sanity check (local, does not publish)

- [ ] `python -m build` succeeds (sdist + wheel)
- [ ] `twine check dist/*` passes
- [ ] Fresh-venv smoke test:
      ```bash
      python -m venv /tmp/earlyon-rc && /tmp/earlyon-rc/bin/pip install dist/*.whl
      /tmp/earlyon-rc/bin/python -c "import earlyon; print(earlyon.__version__)"
      /tmp/earlyon-rc/bin/earlyon --help
      ```
- [ ] Wheel contents look right: `unzip -l dist/*.whl` — package modules
      present, no tests/, no checkpoints, no `htmlcov/`

## 4. Tag and release

- [ ] Commit the version bump: `git commit -am "chore: release vX.Y.Z"`
- [ ] Tag: `git tag vX.Y.Z && git push origin main vX.Y.Z`
- [ ] Watch `release.yml` in the Actions tab until it finishes green

## 5. Post-release verification

- [ ] Package visible on <https://pypi.org/project/earlyon/> at the new version
- [ ] Clean install from PyPI works:
      `python -m venv /tmp/earlyon-pypi && /tmp/earlyon-pypi/bin/pip install earlyon==X.Y.Z`
- [ ] Import + CLI smoke test from that venv (same commands as §3)
- [ ] Create the GitHub release from the tag, pasting the CHANGELOG section:
      `gh release create vX.Y.Z --notes-file <(sed -n '/X.Y.Z/,/^## /p' CHANGELOG.md)`

## If something went wrong

- **Bad artifact published:** `twine yank` (or yank via the PyPI UI), fix,
  release a new patch version. Never delete-and-reupload the same version.
- **Workflow failed after tag push:** fix the workflow on `main`, delete the
  tag locally and remotely only if nothing was published
  (`git tag -d vX.Y.Z && git push origin :refs/tags/vX.Y.Z`), re-tag.
