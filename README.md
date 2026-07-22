<p align="center">
  <img src="assets/banner.svg" alt="earlyon — early-exit inference for PyTorch CV models" width="100%">
</p>

<p align="center">
  <a href="https://github.com/sohams25/earlyon/actions/workflows/ci.yml"><img alt="ci" src="https://github.com/sohams25/earlyon/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.10+-blue">
  <img alt="coverage" src="https://img.shields.io/badge/coverage-95%25-brightgreen">
  <img alt="mypy" src="https://img.shields.io/badge/mypy-strict-blue">
  <img alt="version" src="https://img.shields.io/badge/version-0.2.0-lightgrey">
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

> **`earlyon`** — early exits for PyTorch CV models. Small classifier heads
> partway through the network let easy images stop computing once an exit is
> confident; only hard images run every layer. Attaches to your existing
> backbone via forward hooks. No rewrite, weights load unchanged.

---

```python
import torch
from earlyon.models import resnet50_ee

# pretrained=True fetches ResNet50 ImageNet weights once (~100 MB)
model = resnet50_ee(num_classes=10, pretrained=True).eval()
result = model(torch.randn(1, 3, 224, 224), mode="inference")

result.exit_taken                           # which head answered; -1 = full network
result.estimated_backbone_flops_fraction    # static estimate, not a measurement
result.confidence                           # softmax confidence where it answered
```

earlyon is a research-to-deployment toolkit. Its primary target is batch-1,
latency-sensitive inference; batched routing exists but is conservative.

<p align="center">
  <img src="assets/demo.svg" alt="earlyon demo: an easy image exits at the first head using 12% of the FLOPs, a hard image runs the whole network" width="100%">
</p>

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
of just the top class. Whether an exit may fire at all is an explicit
per-exit boolean (`enabled_exits`) — a disabled exit can never fire, not
even at softmax confidence exactly 1.0. Calibration and
`save_wrapper`/`load_wrapper` are policy-aware, so a calibrated entropy
model reloads as one; checkpoints carry a versioned schema (v2) and older
files migrate with a warning.

Training is two-stage by default: train the backbone exactly as you already
do, freeze it (parameters and BatchNorm stats), then train the small heads
for a few epochs. `joint_train_backbone_and_exits` does end-to-end training
instead when you want the last bit of accuracy and have the budget.
Temperature scaling (Guo et al. 2017) can be fit before calibration to fix
the usual softmax over-confidence; pass `fit_temperature=True` and each head
— every exit and the final classifier — gets its own fitted temperature.
The full split discipline (what may touch which data) is written down in
[`docs/CALIBRATION_AND_BENCHMARK_CONTRACT.md`](docs/CALIBRATION_AND_BENCHMARK_CONTRACT.md).

## Measured results (legacy: earlyon v0.2 methodology)

CIFAR-10, trained on an RTX 4050 Laptop GPU (6 GB) from ImageNet-pretrained
weights: 3–4 epochs for the backbone, 3–4 for the exit heads, thresholds
calibrated to a 1% accuracy budget. **These runs predate v0.3**: they used a
single shared temperature and the pre-fair-runner benchmark; treat them as
indicative, not as v0.3 results. Regenerate with
`python scripts/run_benchmarks.py` (records land in
`docs/benchmarks.json` under `runs`; these live under `legacy_v0_2`).

| Model       | Test Acc | Baseline Acc | Est. FLOPs fraction | % exited early |
|-------------|---------:|-------------:|--------------------:|---------------:|
| ResNet18    |   94.42% |       96.32% |              89.88% |          35.3% |
| ResNet50    |   95.88% |       97.60% |              81.42% |          58.2% |
| MobileNetV2 |   93.31% |       95.08% |              93.90% |           8.5% |

The FLOPs fraction is a per-image *estimate* over the real test set — a
static analysis of the backbone that excludes the exit heads' own small cost
and all routing overhead. For ResNet50 it reads as ~19% of estimated compute
gone for a 1.7% accuracy cost, with 58% of images never reaching `layer4`.
An estimated FLOPs saving is not a latency claim: wall-clock speedup is
hardware- and input-dependent; see the
[latency appendix](#appendix-wall-clock-latency-legacy) before quoting one.
When judging a deployment, also benchmark a smaller static model at matched
accuracy (pass it as a third entry to `benchmark_models`) — if a plain
ResNet18 matches your routed ResNet50, ship the ResNet18.

The MobileNetV2 row is a loss, and it's in the table anyway. An
already-compressed backbone leaves little for early exit to skim: only 8.5%
of images leave early, so the wrapper still does ~94% of the work. Rule of
thumb from these runs: the deeper and heavier the backbone, the more there is
to save.

Reproduce with `python scripts/run_benchmarks.py` (trains and benchmarks) or
`python scripts/re_evaluate.py` (from checkpoints). Raw numbers live in
[`docs/benchmarks.json`](docs/benchmarks.json).

## Measured results (v0.3 fair runner — bounded evidence run)

One seeded, deliberately small validation run (2 epochs per stage, RTX 4050
Laptop GPU, CIFAR-10 at 224px, disjoint train/temperature/calibration/test
splits, identical samples and boundaries for all three models). Full data
and honest interpretation: [`docs/evidence/CUDA_EVIDENCE.md`](docs/evidence/CUDA_EVIDENCE.md).

| Model | Test acc | p50 | p95 | Throughput | Est. FLOPs fraction |
|---|---:|---:|---:|---:|---:|
| ResNet-18 backbone | 94.07% | 1.33 ms | 1.51 ms | 712 ips | 1.00 |
| ResNet-18 early-exit | 92.92% | 1.43 ms | 1.46 ms | 782 ips (**1.10×**) | 0.88 (estimate) |
| MobileNetV2 static baseline | 92.67% | 1.40 ms | 1.48 ms | 702 ips (0.99×) | 1.00 |

Read the negative parts too: the early-exit **median** latency is worse than
the backbone's (routing overhead on the 70% of images that don't exit); the
gain is in throughput and the tail (p95/p99), and the static baseline is
competitive at this training budget. That's the honest shape of early exit
on a fast GPU — reproduce with `python scripts/evidence_run.py`.

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

The same pipeline from the shell, wrap to benchmark:

```bash
earlyon wrap --backbone resnet50 --num-classes 10 --output model.pth
earlyon train backbone --backbone resnet50 --num-classes 10 --dataset cifar10 --output backbone.pth
earlyon train exits --model backbone.pth --dataset cifar10 --output ee.pth
earlyon calibrate --model ee.pth --target-drop 0.01 --output calibrated.pth
earlyon benchmark --model calibrated.pth --device cuda
```

Every command takes `--help`. The rest of the toolkit: `train joint`,
`calibrate --target-compute`, `analyze` (per-exit accuracy), `profile`
(Jetson power), `export` (ONNX).

## Architecture

```
            input (batch 1)
              │
   ┌──────────▼──────────┐   hook    ┌────────────┐  softmax(logits / T_e0)
   │ backbone stage 0..k │──────────▶│ exit head 0│──┐ enabled? confidence ≥ thr?
   └──────────┬──────────┘           └────────────┘  │ yes → return logits, STOP
              │ (only if exit 0 didn't fire)         │ no  ▼ continue
   ┌──────────▼──────────┐   hook    ┌────────────┐  │
   │ backbone stage k..m │──────────▶│ exit head 1│──┤  each head has its OWN
   └──────────┬──────────┘           └────────────┘  │  temperature T and its
              │                                      │  own enabled flag
   ┌──────────▼──────────┐                           │
   │ rest of backbone    │───────▶ final classifier ─┘  (T_final)
   └─────────────────────┘

   training mode: every head produces logits (no routing) → weighted CE loss
   calibration:   collect logits once → fit T per head → greedy threshold grid
   deployment:    eager wrapper (above), or staged stages (docs/STAGED_DEPLOYMENT.md)
```

Three concerns, kept separate: **training** (two-stage by default: your
backbone recipe untouched, then small heads on a frozen backbone),
**calibration** (per-head temperatures + threshold/enablement search on held-
out splits), and **routing** (pure inference-time policy driven by the
config). The full contract is in
[`docs/CALIBRATION_AND_BENCHMARK_CONTRACT.md`](docs/CALIBRATION_AND_BENCHMARK_CONTRACT.md).

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

- `forward(mode="inference")` is batch-size 1 — the latency-sensitive edge
  case earlyon targets. `forward_inference_batched` handles batches, but
  conservatively (see above); masked per-sample routing inside a batch is on
  the roadmap.
- Eager routing has real overhead: every enabled exit evaluates its head and
  synchronises the host (`.item()`) to decide. On a GPU that synchronisation
  can cost more than the skipped layers save, particularly on small
  backbones — theoretical FLOP savings do not guarantee latency savings.
  The fair runner reports both so you can see it.
- `torch.compile` cannot trace the routing control flow. The wrapper raises a
  clear error instead of silently falling back; compile the raw backbone if
  you need it.
- ONNX export (`export_all_exits_to_onnx`) writes a static graph that
  computes **every** exit and leaves routing to the caller — it does not
  short-circuit compute. For a deployable split that genuinely skips later
  stages, see [`docs/STAGED_DEPLOYMENT.md`](docs/STAGED_DEPLOYMENT.md)
  (reference implementation for Sequential backbones).
- `computation_used` / `estimated_backbone_flops_fraction` is a static
  estimate: exit-head cost and routing overhead excluded; backbones that
  reuse modules degrade to a warned low-confidence uniform estimate.
- Compute budgets from `calibrate_thresholds_for_budget` hold as an average
  over the calibration distribution. Nothing caps FLOPs per sample.
- Benchmarks are CIFAR-10 on a laptop GPU, and the published table predates
  the v0.3 fair runner (see the legacy labels). ImageNet-scale numbers and a
  real Jetson table don't exist yet; treat any wall-clock claim accordingly.

## Checkpoint migration

Checkpoints are versioned (`format_version: 2`). Files written by earlyon
≤ 0.2 load with an automatic, deterministic migration (scalar temperature
broadcast per head; legacy "disabled" threshold sentinels become explicit
`enabled_exits=False`) and a warning describing what changed — pinned
against a genuine v0.2-written fixture in the test suite. Details and code
changes: [`docs/MIGRATION.md`](docs/MIGRATION.md).

## Reproducibility

- `pytest -m "not gpu"` — full CPU suite; `pytest -m gpu` on a CUDA host.
- `python scripts/smoke_test.py` — post-install smoke against an installed
  wheel (run it from outside the repo).
- `python scripts/evidence_run.py` — the bounded, seeded CIFAR-10 evidence
  experiment (CUDA; explicit train/temperature/calibration/test splits,
  epoch caps, wall-clock budget). Output lands in `docs/evidence/`.
- `python scripts/run_benchmarks.py` — the longer train+calibrate+benchmark
  pipeline that regenerates `docs/benchmarks.json` (`runs` key).
- Jetson: the exact procedure (tegrastats profiling, per-stage TensorRT) is
  in [`docs/STAGED_DEPLOYMENT.md`](docs/STAGED_DEPLOYMENT.md); no Jetson
  numbers are published because none have been measured.

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

## Appendix: wall-clock latency (legacy)

**Legacy numbers**: measured with the pre-v0.3 runner, which benchmarked the
wrapper and backbone on *different* random inputs — not methodologically
comparable to `benchmark_models` results, which feed every compared model
the identical sample sequence. Kept for transparency until re-measured.

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
