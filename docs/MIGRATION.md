# Migrating to earlyon 0.3

0.3 changes calibration/routing semantics. Old checkpoints load and migrate
automatically; code that touched the changed fields needs the edits below.

## Checkpoints (v1 → v2): automatic

`load_wrapper` detects unversioned (≤0.2) files and migrates them
deterministically, emitting a `UserWarning` describing what changed:

| v1 field | v2 result |
|---|---|
| scalar `temperature: T` | `temperatures = {every exit: T, "final": T}` |
| confidence threshold `1.0` (active policy) | that exit's `enabled_exits[i] = False` |
| entropy threshold `0.0` (active policy) | that exit's `enabled_exits[i] = False` |
| non-finite/non-positive `temperature` | `1.0` (uncalibrated) + warning |

Behavioral note: under v1, a "disabled" (threshold-1.0) exit could still fire
when float32 softmax saturated at exactly 1.0. After migration it is strictly
disabled. If you relied on that edge case, re-enable the exit and set a real
threshold.

Re-saving a migrated model writes `format_version: 2`; v2 files round-trip
with no warnings. Files with a *newer* format version are rejected with an
upgrade message. This is pinned against a genuine v0.2-written fixture in
`tests/test_release_audit.py`.

`custom_ee` checkpoints cannot be rebuilt from a string — pass a factory:

```python
model = load_wrapper("ckpt.pth", factory=lambda: custom_ee(build_backbone(), ...))
```

## Code changes

- **Temperature:** read/write `config.temperatures` (a dict keyed by exit
  name plus `"final"`), not `config.temperature`. The scalar is still
  accepted by the constructor (broadcast once); mutating it after
  construction has no effect on routing.
- **Disabling an exit:** set `config.enabled_exits[i] = False`. Do not use
  threshold sentinels; `validate()` requires thresholds in-range.
- **Compute number:** `result.computation_used` still reads, but the field
  is `estimated_backbone_flops_fraction` — and it is an estimate (see
  `docs/CALIBRATION_AND_BENCHMARK_CONTRACT.md`).
- **CalibrationResult:** `fitted_temperature` is now the final head's fit
  (deprecated); use `result.temperatures` / `result.temperature_fits`.
  New fields: `enabled_exits`, `exit_distribution`, `num_samples`,
  `objective`, `method`, `accuracy_delta`, `schema_version`.
- **Benchmarking:** prefer `benchmark_models({...}, loader=...)`; the old
  single-model helpers still work and now share the fair measurement core.
  `BenchmarkResult.avg_computation_used` → `avg_estimated_flops_fraction`
  (alias kept).
- **Invalid config now raises** (e.g. negative temperature, out-of-range
  threshold) from `EarlyExitConfig.validate()` instead of being silently
  clamped at inference time.
