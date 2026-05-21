# earlyon

[![ci](https://github.com/sohams25/earlyon/actions/workflows/ci.yml/badge.svg)](https://github.com/sohams25/earlyon/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/earlyon/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

production-ready early exit for PyTorch CV models.

```bash
pip install earlyon
```

deep neural networks process every input through every layer. for "easy"
inputs that's wasteful: research has shown 1.3-2.5x speedup at single-sample
edge inference is possible by exiting at intermediate layers when the
network is already confident.

**earlyon** wraps standard torchvision backbones with lightweight exit
heads, ships a two-stage trainer, a greedy threshold calibrator, and
benchmarks (including NVIDIA Jetson power and thermal profiling).

## results on cifar-10

Trained on an RTX 4050 Laptop GPU (6 GB), starting from ImageNet pretrained
weights and fine-tuning for 3-4 stage-1 epochs + 3-4 stage-2 epochs.
Thresholds calibrated with a 1% target accuracy drop.

| Model       | Test Acc | Baseline Acc | Avg FLOPs Used | % samples that exited early | Backbone p50 latency | Wrapper p50 latency* |
|-------------|---------:|-------------:|---------------:|----------------------------:|---------------------:|---------------------:|
| ResNet18    |   94.42% |       96.32% |         89.88% |                       35.3% |              1.32 ms |              0.51 ms |
| ResNet50    |   95.88% |       97.60% |         81.42% |                       58.2% |              2.81 ms |              3.03 ms |
| MobileNetV2 |   93.31% |       95.08% |         93.90% |                        8.5% |               TBD ms |               TBD ms |

*Wrapper latency measured with random-noise input, which can trigger
spurious early exits in trained heads — treat as a best-case rather than a
faithful production number. The honest signal is the **Avg FLOPs Used**
column: this is what the wrapper actually skipped on real test images.

### per-exit distribution on the test set

| Model       | exit_0 (early) | exit_1 | exit_2 | final (no exit) |
|-------------|---------------:|-------:|-------:|----------------:|
| ResNet18    |           9.5% |  25.8% |    n/a |           64.7% |
| ResNet50    |           5.8% |  10.0% |  42.5% |           41.8% |
| MobileNetV2 |           8.0% |   0.4% |    n/a |           91.6% |

### honest observations

- **earlyon is a real win when the architecture has expensive deep layers
  and the dataset has a mix of easy/hard inputs.** ResNet50 saves ~19% of
  FLOPs on CIFAR-10 with only 1.7% accuracy drop, and 58% of test images
  never hit `layer4`.
- **earlyon is barely worth it for MobileNetV2 on CIFAR-10** with this
  training schedule: only 8% of inputs exit early, so the wrapper is doing
  ~94% of the work the backbone does, with a small overhead per layer.
- **wall-clock speedup is not 1.0/avg_comp.** Hook overhead, kernel-launch
  latency, and small per-exit head FLOPs eat some of the theoretical
  savings on small models / small batches. The architect-led review
  predicted this; the numbers above confirm it. The wrapper's value is
  highest on big models with expensive trailing layers.

Reproduce with:
```bash
python scripts/run_benchmarks.py             # train + bench all three
python scripts/re_evaluate.py                # re-eval from saved checkpoints
```

Raw JSON for every run lives at [`docs/benchmarks.json`](docs/benchmarks.json).

Jetson rows: TBD until run on real Jetson hardware. The reproducible script
is `examples/02_jetson_deployment.py`.

## quick start

```python
import torch
from earlyon.models import resnet50_ee

model = resnet50_ee(num_classes=10, pretrained=True)
# train backbone normally, then train exit heads with backbone frozen
# (see examples/01_train_resnet50_cifar10.py)

model.eval()
x = torch.randn(1, 3, 224, 224)
result = model(x, mode="inference")
print(result.exit_taken)        # which exit fired (-1 = final classifier)
print(result.confidence)        # softmax max at that exit
print(result.computation_used)  # fraction of total FLOPs actually run
```

Batched inference (v0.2 — all samples in the batch exit together):

```python
result = model.forward_inference_batched(x_batch)  # x_batch shape (N, 3, H, W)
print(result.exit_taken)              # scalar: the layer everyone exited at
print(result.per_sample_confidence)   # tensor (N,) of per-sample confidences
```

## CLI

```bash
# build a fresh wrapper
earlyon wrap --backbone resnet50 --num-classes 10 --output model.pth

# two-stage training
earlyon train backbone --backbone resnet50 --num-classes 10 \
    --dataset cifar10 --epochs 90 --output backbone.pth
earlyon train exits --model backbone.pth --dataset cifar10 \
    --epochs 20 --output ee.pth

# calibrate confidence thresholds
earlyon calibrate --model ee.pth --target-drop 0.01 --output calibrated.pth

# benchmark throughput
earlyon benchmark --model calibrated.pth --device cuda --runs 500

# jetson power + thermal profile
earlyon profile --model calibrated.pth --runs 200

# per-exit accuracy + class-wise exit distribution
earlyon analyze --model calibrated.pth
```

## how it works

1. **forward hooks** attach early exit heads to specified backbone layers
   (e.g. `layer1`, `layer2`, `layer3` for ResNet, `features.3` and
   `features.10` for MobileNetV2). this avoids rewriting backbone forwards.
2. **single-sample inference** computes softmax at each exit. if confidence
   exceeds the per-exit threshold, the wrapper raises a sentinel exception
   inside the hook to short-circuit the rest of the backbone — this is the
   only reliable way to skip downstream layers from inside a hook.
3. **two-stage training** trains the backbone as a standard classifier
   first, then freezes it (parameters and BatchNorm running stats both),
   and trains only the lightweight exit heads. simpler than joint training
   and avoids gradient conflict.
4. **temperature scaling** (Guo et al. 2017) is fit post-hoc on a held-out
   set before threshold calibration. modern CNNs are systematically
   over-confident; a single scalar divides logits before softmax and
   materially improves the speedup-at-fixed-accuracy curve.
5. **greedy threshold calibration** sweeps a grid per exit (coordinate
   descent, not joint-optimal) and keeps the lowest threshold that holds
   accuracy within the target drop on the validation set.

## current limitations

- **batch size 1** for single-sample inference (`forward(mode="inference")`).
  use `forward_inference_batched(x)` for per-batch routing: all samples in
  the batch exit together at the earliest layer where every sample meets
  its threshold.
- **no `torch.compile` on the inference path.** the conditional control
  flow is incompatible; the wrapper raises a clear error if you try.
  compile the raw backbone instead.
- **ONNX export deferred to v0.2.** the dynamic control flow in
  `_forward_inference` is rejected by torch 2.x's new exporter. tracked
  in `earlyon/onnx.py`.
- **only confidence routing.** entropy and compute-budget routing are
  in the literature but were cut from v0.1 to keep the API small.

## acknowledgements

- BranchyNet (Teerapittayanon et al. 2016) and the ACM 2024 early-exit
  survey (10.1145/3698767) for the foundational ideas
- torchvision for the pretrained backbones (BSD license; earlyon itself
  is MIT)
- fvcore for FLOPs accounting

## license

MIT — see [LICENSE](LICENSE).
