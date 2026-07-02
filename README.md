<p align="center">
  <img src="assets/banner.svg" alt="earlyon — early-exit inference for PyTorch CV models" width="100%">
</p>

<p align="center">
  <a href="https://github.com/sohams25/earlyon/actions/workflows/ci.yml"><img alt="ci" src="https://github.com/sohams25/earlyon/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.10+-49b6ff?style=flat-square&labelColor=0a0a0e">
  <img alt="coverage" src="https://img.shields.io/badge/coverage-96%25-54d18a?style=flat-square&labelColor=0a0a0e">
  <img alt="mypy" src="https://img.shields.io/badge/mypy-strict-5E6AD2?style=flat-square&labelColor=0a0a0e">
  <img alt="version" src="https://img.shields.io/badge/version-0.2.0-FF6B35?style=flat-square&labelColor=0a0a0e">
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-MIT-ECEDF1?style=flat-square&labelColor=0a0a0e"></a>
</p>

> **`earlyon`** — early exits for PyTorch CV models. Small classifier heads
> partway through the network let easy images stop computing once an exit is
> confident; only hard images run every layer. Attaches to your existing
> backbone via forward hooks. No rewrite, weights load unchanged.

---

## The 10-second pitch

```python
import torch
from earlyon.models import resnet50_ee

# pretrained=True fetches ResNet50 ImageNet weights once (~100 MB)
model = resnet50_ee(num_classes=10, pretrained=True).eval()
result = model(torch.randn(1, 3, 224, 224), mode="inference")

result.exit_taken        # which head answered; -1 means the full network ran
result.computation_used  # fraction of the network's FLOPs actually spent
result.confidence        # softmax confidence at the point it answered
```

<p align="center">
  <img src="assets/demo.svg" alt="earlyon demo: an easy image exits at the first head using 12% of the FLOPs, a hard image runs the whole network" width="100%">
</p>

One wrapper call, three numbers per inference: which head answered, what
fraction of the compute ran, and how sure the model was when it stopped.

## Install

```bash
pip install earlyon
```

Or from source for development:

```bash
git clone https://github.com/sohams25/earlyon.git
cd earlyon
pip install -e ".[dev]"
```

Python 3.10+, torch 2.0+. `pretrained=True` downloads torchvision weights on
first use; the CIFAR-10 CLI commands download the dataset on first run.

## Why the full forward pass hurts

A deep network spends the same compute on every image. A frontal close-up of
a car and a blurry, half-occluded bird both pay for all fifty layers, even
though the car is decided after a handful of them. At batch size 1 on an
edge device, that flat cost is your latency and your power budget.

The fix has been in the literature for years: BranchyNet in 2016, then eight
years of follow-ups collected in a 2024 ACM survey (10.1145/3698767) showing
1.3 to 2.5× compute savings at single-sample edge inference. What was
missing is the tool. Each paper ships custom code for one architecture;
earlyon is the `pip install` version.

## What it does

Six ready-made factories, or wrap anything:

| Factory | Backbone | Exits |
|---|---|---|
| `resnet18_ee` | torchvision ResNet18 | 2 (after `layer2`, `layer3`) |
| `resnet50_ee` | torchvision ResNet50 | 3 (after `layer1`, `layer2`, `layer3`) |
| `mobilenetv2_ee` | torchvision MobileNetV2 | 2 (`features.3`, `features.10`) |
| `efficientnet_b0_ee` | torchvision EfficientNet-B0 | 2 (`features.3`, `features.5`) |
| `cifar_resnet_ee` | CIFAR-native ResNet (He et al. 2015) | 3 (3×3 stem, no maxpool, native 32×32) |
| `vit_b_16_ee` | torchvision ViT-B/16 | 2 (after encoder blocks 3 and 9) |

`custom_ee` attaches exits at named layers of any `nn.Module` and infers each
head's width from one dry-run forward. Conv (4D) and transformer token (3D)
features both work; the backbone must already return `(B, num_classes)`
logits.

```python
from earlyon.models import custom_ee

model = custom_ee(backbone, exit_layers=["layer2", "layer3"], num_classes=10)
```

Routing has two policies. `"confidence"` (default) exits when
`softmax(logits).max() >= threshold`; `"entropy"` exits when the softmax
entropy drops below a threshold, which reads the whole distribution instead
of just the top class. Calibration and `save_wrapper`/`load_wrapper` are
policy-aware, so a calibrated entropy model reloads as one.

Training is two-stage by default: train the backbone exactly as you already
do, freeze it (parameters and BatchNorm stats), then train the small heads
for a few epochs. `joint_train_backbone_and_exits` does end-to-end training
instead when you want the last bit of accuracy and have the budget.
Temperature scaling (Guo et al. 2017) can be fit before calibration to fix
the usual softmax over-confidence; pass `fit_temperature=True`.

## Measured results

CIFAR-10, trained on an RTX 4050 Laptop GPU (6 GB) from ImageNet-pretrained
weights: 3–4 epochs for the backbone, 3–4 for the exit heads, thresholds
calibrated to a 1% accuracy budget.

| Model       | Test Acc | Baseline Acc | Avg FLOPs Used | % exited early |
|-------------|---------:|-------------:|---------------:|---------------:|
| ResNet18    |   94.42% |       96.32% |         89.88% |          35.3% |
| ResNet50    |   95.88% |       97.60% |         81.42% |          58.2% |
| MobileNetV2 |   93.31% |       95.08% |         93.90% |           8.5% |

Avg FLOPs Used is measured on the real test set, per image. It is the number
I'd want to see, so it's the one the tool reports. For ResNet50 that means
~19% of the compute gone for a 1.7% accuracy cost, with 58% of images never
reaching `layer4`. Wall-clock speedup is hardware- and input-dependent; see
the [latency appendix](#appendix-wall-clock-latency) before quoting one.

The MobileNetV2 row is a loss, and it's in the table anyway. An
already-compressed backbone leaves little for early exit to skim: only 8.5%
of images leave early, so the wrapper still does ~94% of the work. Rule of
thumb from these runs: the deeper and heavier the backbone, the more there is
to save.

Reproduce with `python scripts/run_benchmarks.py` (trains and benchmarks) or
`python scripts/re_evaluate.py` (from checkpoints). Raw numbers live in
[`docs/benchmarks.json`](docs/benchmarks.json).

## How it compares

Compression makes the model cheaper for every input; early exit spends
compute per input. They stack.

|  | earlyon | per-paper research code | pruning / distillation |
|---|---|---|---|
| works on your existing backbone | yes, via hooks | one architecture each | retrain required |
| adapts compute per input | yes | yes, for that model | no, fixed cost |
| full accuracy still reachable | yes, hard inputs run everything | yes | no, capacity is gone |
| pip install, tests, CI | yes | rarely | yes, mature tools |

If you already prune or distill, wrap the compressed model and take both
savings.

## Usage

This block runs as-is on CPU (synthetic data standing in for your loaders):

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

from earlyon.core.thresholds import calibrate_thresholds
from earlyon.models import resnet18_ee

model = resnet18_ee(num_classes=10, pretrained=False)

# your validation split goes here
val = DataLoader(
    TensorDataset(torch.randn(64, 3, 224, 224), torch.randint(0, 10, (64,))),
    batch_size=8,
)

# 1. train the backbone with your usual recipe, or start from a checkpoint
# 2. freeze it and train the small exit heads
#    (both steps: examples/01_train_resnet50_cifar10.py)

# 3. pick thresholds that keep accuracy within 1% of the full network
calibrate_thresholds(model, val, target_accuracy_drop=0.01)

# 4. deploy
model.eval()
result = model(torch.randn(1, 3, 224, 224), mode="inference")
print(result.exit_taken, result.confidence, result.computation_used)
```

The inference path runs under `torch.inference_mode()` internally, so the
returned prediction carries no autograd graph and is safe in a server loop.

You can also calibrate from the other direction: state a compute budget and
keep as much accuracy as it allows.

```python
from earlyon.core.thresholds import calibrate_thresholds_for_budget

result = calibrate_thresholds_for_budget(model, val, target_computation=0.8)
result.budget_met            # False (plus a warning) if 0.8 is unreachable
result.avg_computation_used  # measured on the calibration set
```

An unreachable budget warns and reports `budget_met=False` rather than
pretending. Budgets hold as an average over the calibration distribution, not
per sample; the [limitations](#limitations) section spells that out.

Batched inference routes conservatively: the whole batch exits at the
earliest layer every sample clears, so the hardest sample sets the pace.

```python
x_batch = torch.randn(16, 3, 224, 224)
result = model.forward_inference_batched(x_batch)
result.exit_taken              # the layer the whole batch left at
result.per_sample_confidence   # tensor of shape (16,)
```

The same pipeline from the shell:

```bash
# wrap a backbone (add --no-pretrained to skip the weight download)
earlyon wrap --backbone resnet50 --num-classes 10 --output model.pth
earlyon wrap --backbone cifar_resnet20 --num-classes 10 --output cifar.pth

# two-stage training on CIFAR-10 (downloads the dataset on first run)
earlyon train backbone --backbone resnet50 --num-classes 10 \
    --dataset cifar10 --epochs 90 --output backbone.pth
earlyon train exits --model backbone.pth --dataset cifar10 --epochs 20 --output ee.pth
earlyon train joint --model backbone.pth --dataset cifar10 --epochs 30 --output joint.pth

# calibrate: accuracy budget, or compute budget
earlyon calibrate --model ee.pth --target-drop 0.01 --output calibrated.pth
earlyon calibrate --model ee.pth --target-compute 0.8 --output budget.pth

# measure
earlyon benchmark --model calibrated.pth --device cuda --runs 500
earlyon profile   --model calibrated.pth --runs 200   # Jetson power + thermals
earlyon analyze   --model calibrated.pth              # per-exit accuracy + distribution
earlyon export    --model calibrated.pth --output model.onnx
```

## How it works

1. Forward hooks attach the exit heads at the named layers, so the backbone
   forward is never rewritten.
2. At inference, each hook checks its policy's criterion. When one passes, a
   sentinel exception short-circuits the rest of the backbone (the only
   reliable way to stop an opaque `forward` from inside a hook).
3. Training freezes the backbone and trains heads (two-stage), or trains
   everything jointly.
4. Temperature scaling is fit before calibration when requested.
5. Calibration collects every head's logits in one batched pass, then sweeps
   a per-exit threshold grid against your accuracy or compute budget in
   tensor math, so the network runs once instead of once per grid point.

The reasoning for each choice, including the ugly parts, is in
[`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md).

## Limitations

- `forward(mode="inference")` is batch-size 1. `forward_inference_batched`
  handles batches, but conservatively (see above); masked per-sample routing
  inside a batch is on the roadmap.
- `torch.compile` cannot trace the routing control flow. The wrapper raises a
  clear error instead of silently falling back; compile the raw backbone if
  you need it.
- ONNX export writes a static graph that computes every exit and leaves
  routing to the caller. The per-sample compute saving only exists in the
  PyTorch wrapper.
- Compute budgets from `calibrate_thresholds_for_budget` hold as an average
  over the calibration distribution. Nothing caps FLOPs per sample.
- Benchmarks are CIFAR-10 on a laptop GPU. ImageNet-scale numbers and a real
  Jetson table don't exist yet; treat any wall-clock claim accordingly.

## Contributing

Bug reports and benchmark results from your hardware are welcome. Start with
[CONTRIBUTING.md](CONTRIBUTING.md); security issues go through
[SECURITY.md](SECURITY.md).

## Roadmap

- ImageNet-scale benchmark runs, and a measured Jetson Orin table to replace
  the "profile it yourself" answer.
- Masked per-sample routing inside a batch (the v0.3 target).
- Grow the backbone factory list as people ask; `custom_ee` covers the gap
  meanwhile.

## Used By

Using earlyon in a project or paper? Open a PR adding yourself here.

## Citation

If earlyon saves your model some FLOPs, a citation is welcome:

```bibtex
@misc{earlyon,
  author       = {Soham},
  title        = {earlyon: early-exit inference for PyTorch CV models},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/sohams25/earlyon}},
  note         = {Version 0.2.0}
}
```

## Acknowledgements

BranchyNet (Teerapittayanon et al. 2016) and the ACM 2024 early-exit survey
(10.1145/3698767) for the ideas; torchvision (BSD) for the backbones; fvcore
for FLOPs accounting. earlyon itself is MIT.

## Appendix: wall-clock latency

Throughput below uses a random-noise input, which can trigger spurious early
exits, so read it as a best-case bound rather than a claim. RTX 4050 Laptop
GPU, batch 1, 224×224, 50-iteration warmup, 300 iterations.

| Model       | Backbone p50 | Wrapper p50 (noise input) |
|-------------|-------------:|--------------------------:|
| ResNet18    |      1.32 ms |                   0.51 ms |
| ResNet50    |      2.81 ms |                   3.03 ms |
| MobileNetV2 |       TBD ms |                    TBD ms |

Reproducible via `scripts/re_evaluate.py`; raw per-run data in
[`docs/benchmarks.json`](docs/benchmarks.json).

## License

MIT — see [LICENSE](LICENSE).
