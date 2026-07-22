# Test fixtures

- `tiny_models.py` — tiny CNN/transformer backbones for fast tests.
- `v1_cifar_resnet20.pth` (~1.2 MB) — a genuine pre-v0.3 (format v1)
  checkpoint written by the actual historical `save_wrapper` (git `main`,
  commit a5f8414) on a seeded, untrained `cifar_resnet_ee(depth=20)`.
  Contains the legacy scalar `temperature` (1.6) and the v1 "disabled"
  sentinel threshold (confidence 1.0 at exit 1). Used by
  `tests/test_release_audit.py::test_real_v1_fixture_checkpoint_migrates`
  to prove the v1→v2 migration against a real prior-version file, not a
  hand-built dict. Deliberately force-added past the `*.pth` gitignore.
