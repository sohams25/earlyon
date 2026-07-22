# Security review — earlyon 0.3 release candidate

Date: 2026-07-22. Scope: the `earlyon` package, scripts, tests, docs and
packaged artifacts on the v0.3.0 release-readiness branch (merged to
`main` via PR #5).

## Checkpoint trust assumptions

- `load_wrapper` calls `torch.load(..., map_location="cpu", weights_only=True)`.
  With `weights_only=True`, PyTorch's restricted unpickler refuses arbitrary
  object construction, so a malicious `.pth` cannot execute code through the
  earlyon loading path. The payload is then structurally validated
  (`format_version`, config shapes, exit-point cross-check) before any state
  is applied.
- This protects the *loading* step only. A checkpoint still controls model
  *behavior* (weights, thresholds, enablement, temperatures); loading an
  untrusted checkpoint yields an untrusted model. Treat checkpoints like
  data, not like code — but do not deploy models from untrusted sources.
- `save_wrapper` writes only tensors, plain Python scalars/containers and
  strings; no custom classes are pickled.

## Model-code execution assumptions

- `custom_ee`, `EarlyExitWrapper` and `staged_model` execute the
  **user-supplied backbone's `forward`** (dry runs, FLOPs probes, routing).
  Wrapping a model is running its code. earlyon adds no sandboxing and makes
  no claim to; only wrap modules you would run anyway.
- `load_wrapper(..., factory=...)` calls the user-supplied factory — same
  trust level as the caller's own code.
- No use of `eval`, `exec`, dynamic imports of user strings, or `pickle`
  outside `torch.save/load` as described above.

## Dataset and weight provenance

- Datasets: CIFAR-10 via `torchvision.datasets` (checksummed download by
  torchvision) into a gitignored `data/` directory. No datasets are
  committed or redistributed.
- Weights: ImageNet-pretrained weights come from torchvision's official hub
  (HTTPS, hash-pinned filenames) when `pretrained=True`. earlyon
  redistributes no third-party weights. The only committed binary is
  `tests/fixtures/v1_cifar_resnet20.pth` (~1.2 MB): a *seeded, untrained*
  model saved with the historical v1 format purely to test checkpoint
  migration; it contains no trained knowledge and no external data.

## Subprocess boundaries

- The only subprocess is `tegrastats` (Jetson telemetry): fixed argv list,
  no `shell=True`, no user-controlled arguments (the interval is a validated
  integer), stdout parsed with anchored regexes, stderr discarded, process
  terminated/killed with timeouts and pipes closed on stop. Spawn failure
  degrades to "unavailable" rather than raising mid-benchmark.

## Filesystem / temp / network

- Library code performs no network access. Downloads happen only through
  torchvision (datasets/weights) when the user opts in via CLI/scripts.
- Tests use `tmp_path`/`TemporaryDirectory`; no world-writable fixed paths.
- No archive extraction is performed by earlyon itself.

## Scan results (this review)

- No credentials, API keys, tokens or private keys in the tree.
- No private/internal paths, employer identifiers, or non-public material.
  The author contact in `pyproject.toml` is deliberate public metadata.
- Coverage output, datasets, build artifacts (`htmlcov/`, `data/`, `dist/`,
  `.coverage`) are gitignored and untracked.
- The wheel contains only the `earlyon` package; the sdist prunes `tests/`
  (`MANIFEST.in`), so the binary fixture ships in neither artifact.

## Reporting

Vulnerabilities: follow `SECURITY.md` (private disclosure; do not open
public issues for security reports).
