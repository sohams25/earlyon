# changelog

format follows [keep a changelog](https://keepachangelog.com/en/1.1.0/).

## unreleased

### added
- **compute-budget calibration**: `calibrate_thresholds_for_budget` /
  `earlyon calibrate --target-compute`, the mirror of the accuracy-budget
  search. State the average FLOPs fraction the deployed model may use (e.g.
  `0.8`); the greedy sweep finds the thresholds that meet it with the least
  accuracy loss. Policy-aware (confidence and entropy), supports the same
  temperature-scaling options, and reports `budget_met=False` plus a
  `UserWarning` when the budget is unattainable with the model's exit points
  instead of silently missing it. `CalibrationResult` gains
  `target_computation` and `budget_met` fields (defaulted, backward
  compatible).
- **ONNX export**: `export_to_onnx` / `earlyon export` write a portable static
  multi-output graph (one output per exit plus the final classifier). Routing
  stays at runtime: the graph computes every exit and the caller picks the
  first confident one. Works for both conv and transformer exit heads; verified
  against onnxruntime. (`onnxruntime` added to the `dev` extra for testing.)
- **transformer support**: `vit_b_16_ee` wraps torchvision ViT-B/16 with two
  early exits (after encoder blocks 3 and 9). `EarlyExitHead` now accepts 3D
  token features ``(B, N, D)`` (CLS or mean pooling) and 2D vectors in addition
  to 4D conv maps, so earlyon is no longer CNN-only.
- `custom_ee(backbone, exit_layers, num_classes)`: wrap **any** `nn.Module` with
  early exits at named submodules, auto-inferring each exit's feature width from
  a single dry-run forward. Fills the previously-documented-but-missing
  custom-model entry point.
- `vit_b_16` is a recognized CLI/`build_model` backbone (round-trips like the
  other factories); `custom` models raise a clear error in `build_model` since
  an arbitrary backbone can't be reconstructed from a string.
- the three trainers now use ``val_loader``: when supplied they compute
  per-epoch validation loss/accuracy and report them via
  ``TrainStepLog.val_loss``/``val_accuracy`` (previously the parameter was
  accepted but ignored with a warning). Opt-in `earlyon train … --validate` CLI
  flag, and a ``val_batch_size`` argument on ``cifar10_loaders`` for fast
  batched validation.

### fixed
- an empty custom `grid` passed to either calibrator now raises instead of
  silently "calibrating" a model that never exits early.
- `EarlyExitConfig` rejects `num_classes < 2` (softmax confidence over one
  class is always 1.0, so routing degenerates).
- `custom_ee` with a nonexistent exit layer raises a `ValueError` naming the
  available layers instead of torch's raw `AttributeError`.
- `save_wrapper` warns when saving a `custom_ee` model, at save time rather
  than at the failed `load_wrapper` call after training.
- the CLI accepts `--backbone cifar_resnet<depth>` (e.g. `cifar_resnet20`);
  previously the CIFAR-native factory from the models table had no CLI path.
- `earlyon calibrate --target-compute` values outside `(0, 1]` fail at option
  parsing with a clean click error instead of a traceback (and before any
  dataset download starts).

### changed
- threshold calibration evaluates trials against cached exit logits (one
  batched forward pass) instead of re-running the network per grid point at
  batch size 1: ~25× faster on the README quick-start scenario (20.1s →
  0.8s on CPU), byte-identical thresholds, equivalence against the real
  routing path pinned by a test.
- consolidated duplicated definitions: a single ``Batch`` alias and
  ``exit_label`` helper in ``earlyon.core.types``, and one ``identity`` final
  classifier in ``earlyon.models._common`` (were copy-pasted across 5 and 4
  files respectively). No behavior change.

## 0.2.0 - 2026-06-04

### added
- entropy routing policy (`routing_policy="entropy"`): exit when softmax entropy
  falls below a per-exit threshold, alongside the existing confidence policy
- `joint_train_backbone_and_exits` + `earlyon train joint` CLI: train backbone
  and exit heads together as an alternative to two-stage training
- `cifar_resnet_ee`: CIFAR-native ResNet (He et al. 2015, 6n+2 depth) for 32x32
  input without upsampling
- `efficientnet_b0_ee` backbone factory and `--backbone efficientnet_b0` CLI option
- `forward_inference_batched`: conservative per-batch routing (all samples in a
  batch exit together at the earliest layer every sample clears)
- `benchmark_wrapper_on_loader`: real-data throughput on samples from a
  DataLoader, the honest input-distribution signal vs. random-noise input
- `fit_temperature` opt-in in `calibrate_thresholds` (post-hoc temperature
  scaling fit before the threshold search)
- `@pytest.mark.gpu` test suite covering CUDA device placement, the benchmark
  sync path, and the on-device BatchNorm-freeze invariant
- `SECURITY.md` with a private vulnerability-disclosure policy

### fixed
- inference path now runs under `torch.inference_mode()`: `model(x,
  mode="inference")` no longer retains an autograd graph on the returned
  prediction (server memory growth) and no longer emits a requires-grad-to-scalar
  warning, even when the caller forgets `no_grad`
- `calibrate_thresholds` is now routing-policy-aware: calibrating an
  entropy-routed model writes `entropy_thresholds` (the field the router reads)
  instead of being a silent no-op, and respects entropy's inverted monotonicity
- `fit_temperature` iterates LBFGS to convergence (was a single underfitting
  step) and rejects non-finite logits / falls back to T=1.0 on divergence so a
  NaN can no longer poison every downstream softmax
- `save_wrapper`/`load_wrapper` now persist `routing_policy` and
  `entropy_thresholds`; an entropy-routed checkpoint no longer reloads as
  confidence-routed. Pre-0.2 checkpoints still load (fields default), and load
  re-validates the policy (rejecting an unknown policy or entropy-without-thresholds)
- `cifar_resnet_ee` checkpoints now round-trip: `build_model` reconstructs the
  CIFAR-native ResNet from its depth-encoded backbone string (was unloadable)
- calibration uses strictly non-firing search seeds, so a float32-saturated
  softmax can no longer fire at the no-exit baseline and corrupt the threshold
  search (affects both confidence and entropy policies)
- `per_layer_flops` restores the backbone's train/eval mode after the FLOPs
  probe instead of silently leaving it in eval at wrapper construction
- `benchmark_wrapper_on_loader` rejects an empty loader instead of spinning forever
- the inference temperature guard tolerates a non-finite `config.temperature`
  (falls back to 1.0) instead of producing all-NaN softmax probabilities

### changed
- `mypy --strict` is clean (0 errors) and enforced in CI; `isort --check` added
  to CI. Full lint/type/test gate now blocks merges
- trainers warn when a `val_loader` is passed (accepted but not yet used)
- Development Status classifier moved from Alpha to Beta

## 0.1.0 - 2026-05-21

initial release.

### added
- `EarlyExitWrapper` base class with forward-hook-based exit registration
- pre-configured factories: `resnet18_ee`, `resnet50_ee`, `mobilenetv2_ee`
- two-stage trainer: stage 1 trains backbone standalone, stage 2 freezes
  backbone (params and batchnorm running stats) and trains exit heads only
- weighted multi-exit cross-entropy loss
- post-hoc temperature scaling (guo et al. 2017) via lbfgs
- greedy confidence-threshold calibration over a validation set
- throughput / latency benchmark harness with cuda sync and warmup
- jetson power and thermal profiler via non-blocking tegrastats subprocess
- click cli: `wrap`, `train backbone`, `train exits`, `calibrate`,
  `benchmark`, `profile`, `analyze`
- per-thread state in wrapper (safe under shared-instance inference)
- `weights_only=True` on checkpoint load
- two examples: cifar-10 training and jetson deployment

### deliberately out of scope for 0.1
- batched per-sample routing (inference is batch=1)
- onnx export with conditional control flow
- joint training of backbone and exits
- entropy or budget routing
- efficientnet, vit, or other backbones
- tensorrt integration
