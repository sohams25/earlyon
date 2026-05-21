# earlyon backlog

Autonomous cron pops the first unchecked `[ ]` item, completes it, commits,
pushes, and ticks the box. One item per tick. Tier 3 items are reserved for
manual sessions and are ignored by the cron.

Cron tag: `earlyon-backlog-cron-v1`

## tier 1 — small, safe (cron-eligible)

- [x] **flops overcount warning**: in `earlyon/core/flops.py`, log a warning when `running / total > 1.05` before applying the `min(..., 1.0)` cap. Add a unit test that asserts the warning fires on a synthetic overcount case.
- [x] **load_wrapper config length check**: in `earlyon/utils.py:load_wrapper`, validate `len(cfg["confidence_thresholds"])` against `len(model.config.exit_points)` BEFORE mutating any config fields. Raise `ValueError` with a clear message on mismatch. Add a unit test.
- [x] **stage2 accuracy field clarity**: in `earlyon/training/two_stage_trainer.py`, the `accuracy` field of `TrainStepLog` reports exit_0 only. Either rename to `exit0_accuracy` or add a per-exit breakdown. Update tests and the example script.
- [ ] **readme latency footnote move**: move the wrapper-latency column out of the headline results table into a footnote/appendix in `README.md`. The headline table should only show: test acc, baseline acc, avg flops used, % samples exiting early.
- [x] **default_log print comment**: in `earlyon/training/two_stage_trainer.py:_default_log`, add a one-line comment explaining that `print()` is intentional for a user-facing CLI default callback.
- [ ] **code of conduct**: add `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1) at repo root.
- [ ] **security policy**: add `SECURITY.md` at repo root with a simple disclosure address.
- [ ] **github labels**: create labels `bug`, `enhancement`, `good-first-issue`, `v0.2`, `documentation`, `help-wanted` via `gh label create`. Idempotent (skip if exists).
- [ ] **coverage badge**: run `pytest --cov=earlyon --cov-report=term`, capture the percentage, add a coverage badge to the README pointing at a static value (no codecov dependency).

## tier 2 — medium effort (cron-eligible, code-reviewer invoked)

- [ ] **real-data throughput bench**: in `earlyon/benchmarking/throughput.py`, add a `benchmark_wrapper_on_loader` function that takes a DataLoader and runs the bench on real samples instead of random noise. Update `scripts/re_evaluate.py` to use it. Document the new metric in README.
- [ ] **entropy routing policy**: add `routing_policy="entropy"` support. The hook computes entropy on softmax; if entropy < threshold, exit. Add to `EarlyExitConfig` validation. Tests + docs.
- [ ] **joint trainer**: implement `JointTrainer` in `earlyon/training/joint_trainer.py` for users who want backbone+exits trained simultaneously. Counterpart to `TwoStageTrainer`. Tests.
- [ ] **cifar-native resnet variant**: add `cifar_resnet_ee` in `earlyon/models/` — small ResNet (~11M params) designed for 32x32 input, no upsampling needed. Tests + a benchmark entry in scripts.
- [ ] **fit_temperature integration**: wire `fit_temperature` into `calibrate_thresholds` as an opt-in `fit_temperature: bool = False` argument. When true, fit a temperature on the val set before threshold search.

## tier 3 — large (manual only, cron must skip)

- ONNX export with TorchScript `If` op
- True per-sample batched routing (masked execution)
- ViT wrapper
- mkdocs site on GitHub Pages
- PyPI publish (needs user-side trusted-publishing config)
- Real Jetson benchmark numbers (needs hardware)
