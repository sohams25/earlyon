# earlyon

production-ready early exit for PyTorch CV models.

```bash
pip install earlyon
```

deep neural networks process every input through every layer. for "easy"
inputs that's wasteful — research has shown 1.3-2.5x speedup at single-sample
edge inference is possible by exiting at intermediate layers when the
confidence is high enough.

**earlyon** wraps standard torchvision backbones (ResNet18, ResNet50,
MobileNetV2) with lightweight exit heads, ships a two-stage trainer, a
greedy threshold calibrator, and benchmarks (including NVIDIA Jetson power
and thermal profiling).

## the table

| Model       | Dataset  | Device          | Baseline FPS | earlyon FPS | Speedup | Accuracy*    | Avg FLOPs Used |
|-------------|----------|-----------------|--------------|-------------|---------|--------------|----------------|
| ResNet50    | CIFAR-10 | RTX 3060        | TBD          | TBD         | TBD     | TBD (TBD)    | TBD            |
| ResNet50    | CIFAR-10 | Jetson Orin NX  | TBD          | TBD         | TBD     | TBD (TBD)    | TBD            |
| MobileNetV2 | CIFAR-10 | RTX 3060        | TBD          | TBD         | TBD     | TBD (TBD)    | TBD            |
| MobileNetV2 | CIFAR-10 | Jetson Orin NX  | TBD          | TBD         | TBD     | TBD (TBD)    | TBD            |

*\*Accuracy in parentheses = full model (no early exit) baseline. Numbers are
TBD until reproduced on real hardware; the script that fills them in is
`examples/02_jetson_deployment.py`.*

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

- **batch size 1** at inference time. per-sample routing in batched
  execution requires masked compute or recompute; that's a v0.2 problem.
- **no `torch.compile` on the inference path.** the conditional control flow
  falls back to eager mode silently, killing the speedup. compile the
  backbone for stage-1 training if you want; not the wrapped inference.
- **ONNX export** is not in v0.1. the conditional routing maps poorly to
  static graphs; the right answer (`If`-op via TorchScript) needs care and
  is queued for v0.2.
- **only confidence routing.** entropy and compute-budget routing are
  in the literature but were cut from v0.1 to keep the API small.

## acknowledgements

- BranchyNet (Teerapittayanon et al. 2016) and the ACM 2024 early-exit
  survey (10.1145/3698767) for the foundational ideas
- torchvision for the pretrained backbones (BSD license — earlyon itself
  is MIT)
- fvcore for FLOPs accounting

## license

MIT — see [LICENSE](LICENSE).
