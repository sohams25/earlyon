<p align="center">
  <img src="assets/banner.svg" alt="earlyon — early-exit inference for PyTorch CV models" width="100%">
</p>

<p align="center">
  <a href="https://github.com/sohams25/earlyon/actions/workflows/ci.yml"><img src="https://github.com/sohams25/earlyon/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <a href="https://pypi.org/project/earlyon/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="python"></a>
  <img src="https://img.shields.io/badge/coverage-97%25-brightgreen.svg" alt="coverage">
  <img src="https://img.shields.io/badge/mypy-strict-blue.svg" alt="mypy strict">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="license"></a>
</p>

<p align="center"><b>Production-ready early exit for PyTorch CV models.</b></p>

```bash
pip install earlyon
```

Deep neural networks push every input through every layer. For "easy" inputs
that's wasteful: research has shown **1.3–2.5× speedup** at single-sample edge
inference is possible by exiting at intermediate layers once the network is
already confident.

**earlyon** wraps standard torchvision backbones with lightweight exit heads and
ships everything around them: two-stage *and* joint trainers, confidence *and*
entropy routing, a greedy threshold calibrator with optional temperature
scaling, and a benchmark suite (including NVIDIA Jetson power/thermal profiling).

---

## results on cifar-10

Trained on an RTX 4050 Laptop GPU (6 GB), starting from ImageNet pretrained
weights and fine-tuning for 3–4 stage-1 epochs + 3–4 stage-2 epochs.
Thresholds calibrated with a 1% target accuracy drop.

| Model       | Test Acc | Baseline Acc | Avg FLOPs Used | % samples that exited early |
|-------------|---------:|-------------:|---------------:|----------------------------:|
| ResNet18    |   94.42% |       96.32% |         89.88% |                       35.3% |
| ResNet50    |   95.88% |       97.60% |         81.42% |                       58.2% |
| MobileNetV2 |   93.31% |       95.08% |         93.90% |                        8.5% |

The **Avg FLOPs Used** column is the honest signal: it measures the work
actually skipped on real test images. Wall-clock latency depends on hardware
and on whether your input distribution triggers spurious early exits; see the
[latency appendix](#appendix-wall-clock-latency).

### per-exit distribution on the test set

| Model       | exit_0 (early) | exit_1 | exit_2 | final (no exit) |
|-------------|---------------:|-------:|-------:|----------------:|
| ResNet18    |           9.5% |  25.8% |    n/a |           64.7% |
| ResNet50    |           5.8% |  10.0% |  42.5% |           41.8% |
| MobileNetV2 |           8.0% |   0.4% |    n/a |           91.6% |

### honest observations

- **earlyon is a real win when the architecture has expensive deep layers and
  the dataset has a mix of easy/hard inputs.** ResNet50 saves ~19% of FLOPs on
  CIFAR-10 with only a 1.7% accuracy drop, and 58% of test images never hit
  `layer4`.
- **earlyon is barely worth it for MobileNetV2 on CIFAR-10** with this training
  schedule: only 8% of inputs exit early, so the wrapper does ~94% of the work
  the backbone does, with a small per-layer overhead.
- **wall-clock speedup is not 1.0/avg_comp.** Hook overhead, kernel-launch
  latency, and small per-exit head FLOPs eat some of the theoretical savings on
  small models / small batches. The wrapper's value is highest on big models
  with expensive trailing layers.

Reproduce with:
```bash
python scripts/run_benchmarks.py   # train + bench all three
python scripts/re_evaluate.py      # re-eval from saved checkpoints
```

Raw JSON for every run lives at [`docs/benchmarks.json`](docs/benchmarks.json).
Jetson rows are TBD until run on real Jetson hardware; the reproducible script
is `examples/02_jetson_deployment.py`.

---

## quick start

```python
import torch
from earlyon.models import resnet50_ee

model = resnet50_ee(num_classes=10, pretrained=True)
# train backbone normally, then train exit heads with the backbone frozen
# (see examples/01_train_resnet50_cifar10.py)

model.eval()
x = torch.randn(1, 3, 224, 224)
result = model(x, mode="inference")     # runs under torch.inference_mode() internally
print(result.exit_taken)        # which exit fired (-1 = final classifier)
print(result.confidence)        # softmax max at that exit
print(result.computation_used)  # fraction of total FLOPs actually run
```

The inference path runs under `torch.inference_mode()` for you, so the returned
prediction never carries an autograd graph — safe to call in a server loop.

Batched inference (all samples in a batch exit together at the earliest layer
every sample clears):

```python
result = model.forward_inference_batched(x_batch)  # x_batch shape (N, 3, H, W)
print(result.exit_taken)              # scalar: the layer everyone exited at
print(result.per_sample_confidence)   # tensor (N,) of per-sample confidences
```

---

## models

Pre-configured factories — each returns an `EarlyExitWrapper`:

| Factory | Backbone | Exits | Notes |
|---------|----------|-------|-------|
| `resnet18_ee`      | torchvision ResNet18  | 2 | exits after `layer2`, `layer3` |
| `resnet50_ee`      | torchvision ResNet50  | 3 | exits after `layer1`, `layer2`, `layer3` |
| `mobilenetv2_ee`   | torchvision MobileNetV2 | 2 | exits at `features.3`, `features.10` |
| `efficientnet_b0_ee` | torchvision EfficientNet-B0 | 2 | exits at `features.3`, `features.5` |
| `cifar_resnet_ee`  | CIFAR-native ResNet (He et al. 2015) | 3 | 3×3 stem, no maxpool, 32×32 input — no upsampling |

```python
from earlyon.models import resnet18_ee, efficientnet_b0_ee, cifar_resnet_ee

m1 = resnet18_ee(num_classes=10)
m2 = efficientnet_b0_ee(num_classes=100)
m3 = cifar_resnet_ee(num_classes=10, depth=20)   # 6n+2 depth, native 32×32
```

---

## routing policies

Set `routing_policy` on the config to choose how an intermediate head decides to
exit:

- **`"confidence"`** (default) — exit when `softmax(logits).max() >= threshold`.
  Calibrated via `confidence_thresholds`.
- **`"entropy"`** — exit when `H(softmax(logits)) <= threshold` (low entropy =
  high certainty). Calibrated via `entropy_thresholds`.

`calibrate_thresholds` is policy-aware: it calibrates whichever list the active
policy reads. `save_wrapper`/`load_wrapper` round-trip the policy and both
threshold lists, so a calibrated entropy model reloads as an entropy model.

```python
from earlyon.core.thresholds import calibrate_thresholds

result = calibrate_thresholds(model, val_loader, target_accuracy_drop=0.01,
                              fit_temperature=True, temperature_loader=cal_loader)
print(result.policy, result.thresholds, result.fitted_temperature)
```

---

## training strategies

Two recipes, same wrapper:

- **two-stage** (recommended) — train the backbone as a standard classifier,
  then freeze it (parameters *and* BatchNorm running stats) and train only the
  lightweight exit heads. No gradient conflict; add exits to any pretrained model.
- **joint** — train backbone and exits together end-to-end with a single
  weighted multi-exit loss. Pick this for absolute peak accuracy when you have
  the compute budget.

```python
from earlyon.training import (
    stage1_train_backbone, stage2_train_exits,   # two-stage
    joint_train_backbone_and_exits,              # joint
)
```

---

## CLI

```bash
# build a fresh wrapper
earlyon wrap --backbone resnet50 --num-classes 10 --output model.pth

# two-stage training
earlyon train backbone --backbone resnet50 --num-classes 10 \
    --dataset cifar10 --epochs 90 --output backbone.pth
earlyon train exits --model backbone.pth --dataset cifar10 --epochs 20 --output ee.pth

# joint training (alternative to two-stage)
earlyon train joint --model backbone.pth --dataset cifar10 --epochs 30 --output joint.pth

# calibrate thresholds (confidence or entropy, per the model's routing_policy)
earlyon calibrate --model ee.pth --target-drop 0.01 --output calibrated.pth

# throughput + latency benchmark
earlyon benchmark --model calibrated.pth --device cuda --runs 500

# jetson power + thermal profile
earlyon profile --model calibrated.pth --runs 200

# per-exit accuracy + class-wise exit distribution
earlyon analyze --model calibrated.pth
```

---

## how it works

1. **forward hooks** attach early-exit heads to specified backbone layers (e.g.
   `layer1`, `layer2`, `layer3` for ResNet; `features.3`, `features.10` for
   MobileNetV2). No backbone forward is rewritten.
2. **single-sample inference** computes softmax at each exit. If the routing
   policy's criterion is met, the wrapper raises a sentinel exception inside the
   hook to short-circuit the rest of the backbone — the only reliable way to
   skip downstream layers from inside a hook. The whole path runs under
   `torch.inference_mode()`.
3. **two-stage / joint training** — train the backbone first then freeze it
   (params and BN stats) and train only the heads, or train everything together.
4. **temperature scaling** (Guo et al. 2017) is fit post-hoc on a held-out set
   before calibration. Modern CNNs are over-confident; one scalar divides logits
   before softmax and materially improves the speedup-at-fixed-accuracy curve.
5. **greedy threshold calibration** sweeps a per-exit grid (coordinate descent)
   and keeps the most aggressive threshold that holds accuracy within the target
   drop on the validation set.

---

## current limitations

- **batch size 1** for single-sample inference (`forward(mode="inference")`).
  Use `forward_inference_batched(x)` for per-batch routing: all samples in the
  batch exit together at the earliest layer every sample clears.
- **no `torch.compile` on the inference path.** The conditional control flow is
  incompatible; the wrapper raises a clear error if you try. Compile the raw
  backbone instead.
- **ONNX export not yet supported.** The dynamic control flow in the inference
  path is rejected by torch 2.x's exporter; deploy from PyTorch directly. Tracked
  in `earlyon/onnx.py`.
- **no compute-budget routing yet.** Confidence and entropy routing ship today;
  budget-constrained routing is on the roadmap.

---

## acknowledgements

- BranchyNet (Teerapittayanon et al. 2016) and the ACM 2024 early-exit survey
  (10.1145/3698767) for the foundational ideas
- torchvision for the pretrained backbones (BSD license; earlyon itself is MIT)
- fvcore for FLOPs accounting

---

## appendix: wall-clock latency

Throughput numbers here use a random-noise dummy input, which can trigger
spurious early exits in trained heads. Treat them as a best-case upper bound
rather than a faithful production number — the honest signal is **Avg FLOPs
Used** in the headline table, measured on the real CIFAR-10 test set.

Measured on RTX 4050 Laptop GPU (6 GB), batch size 1, 224×224 input,
50-iteration warmup, 300 iterations measured.

| Model       | Backbone p50 | Wrapper p50 (noise input) |
|-------------|-------------:|--------------------------:|
| ResNet18    |      1.32 ms |                   0.51 ms |
| ResNet50    |      2.81 ms |                   3.03 ms |
| MobileNetV2 |       TBD ms |                    TBD ms |

Reproducible via `python scripts/re_evaluate.py` against the saved checkpoints
from `scripts/run_benchmarks.py`. Raw per-run measurements in
[`docs/benchmarks.json`](docs/benchmarks.json).

---

## license

MIT, see [LICENSE](LICENSE).
