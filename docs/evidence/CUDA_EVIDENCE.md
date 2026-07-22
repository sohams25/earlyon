# Bounded CUDA evidence run — 2026-07-22

Machine-readable source: [`cuda_evidence.json`](cuda_evidence.json).
Reproduce: `python scripts/evidence_run.py` (seed 42; ~10 min compute on the
hardware below after the CIFAR-10 download).

**Purpose.** A bounded, seeded, end-to-end *system validation* — training →
per-head temperature fit → threshold calibration → held-out evaluation →
fair 3-model benchmark — under the v0.3 methodology. It is deliberately
under-trained (2 epochs per stage) and is **not** a tuned headline
benchmark. Negative and mixed results below are reported as-is.

## Setup

| | |
|---|---|
| Hardware | NVIDIA RTX 4050 Laptop GPU (6 GB), CUDA 12.8 |
| Software | Python 3.13.9, torch 2.10.0+cu128, torchvision 0.25.0, earlyon 0.2.0 (0.3 RC branch) |
| Dataset | CIFAR-10 (torchvision download), images upsampled to 224×224 |
| Splits | train 45,000 · temperature 2,500 · calibration 2,500 · test 10,000 (official) — disjoint; the threshold search never saw test labels |
| Model | `resnet18_ee` (exits after `layer2`, `layer3`; heads: hidden 128, dropout 0.2), ImageNet-pretrained, fine-tuned 2+2 epochs (seed 42) |
| Static baseline | `mobilenet_v2` backbone, ImageNet-pretrained, fine-tuned 2 epochs (batch 32 for memory) |
| Determinism | seed fixes init/data order; bitwise GPU determinism not enforced (cuDNN) |

## Calibration (never touches test data)

Per-head fitted temperatures: e0 = 0.730, e1 = 0.641, final = 0.888 (all
fits converged) — three genuinely different values, which is the point of
per-head fitting. Greedy grid (accuracy budget 1%) selected thresholds
[0.7, 0.9] with both exits enabled. Calibration-split accuracy: 94.44%
(final head only) → 93.60% routed (−0.84%, inside the 1% budget on that
split). Calibration took 4.6 s (one cached-logits pass).

## Held-out test results (10,000 images)

| Model | Test accuracy | Est. backbone FLOPs fraction |
|---|---:|---:|
| ResNet-18 backbone (full) | **94.07%** | 1.000 |
| ResNet-18 early-exit (routed) | 92.92% | 0.880 (estimate; excludes heads/routing) |
| MobileNetV2 static baseline | 92.67% | 1.000 (of its own, smaller network) |

Exit distribution: 5.6% at exit 0, 24.9% at exit 1, 69.5% ran the full
network. The routed accuracy drop on *test* is 1.15% — slightly beyond the
1% budget calibrated on the calibration split. That gap is expected
generalization error of the calibration estimate and is reported, not
hidden.

## Fair benchmark (identical real test samples, model-only boundary, batch 1, 300 runs, 50 warmup, per-iteration CUDA sync)

| Model | p50 | p95 | p99 | Throughput | Accuracy on timed samples |
|---|---:|---:|---:|---:|---:|
| ResNet-18 backbone | 1.33 ms | 1.51 ms | 1.65 ms | 711.9 ips | 93.7% |
| ResNet-18 early-exit | 1.43 ms | 1.46 ms | 1.48 ms | 781.8 ips | 91.7% |
| MobileNetV2 static | 1.40 ms | 1.48 ms | 1.50 ms | 702.1 ips | 93.0% |

Speedups (same samples, same boundary): early-exit vs backbone **1.10×**
throughput; static baseline vs backbone 0.99×. Peak CUDA memory during the
timed comparison: 350 MB. Noise-input variant (best-case bound, spurious
exits possible): 2.05× — reported separately in the JSON and *not* a
practical-speed claim.

## Honest interpretation

- The early-exit model's **median latency is worse** than the raw backbone
  (1.43 vs 1.33 ms) even though throughput is 10% better: when no exit
  fires (70% of images) the wrapper pays the full backbone *plus* two head
  evaluations and host syncs. The win comes from the 30% of images that
  exit early, which also compresses the tail (p95/p99 are better). This is
  exactly the routing-overhead behavior the documentation warns about.
- A ~12% estimated FLOPs saving translated to a ~10% throughput gain on
  this GPU — estimates and wall-clock are close here, but that is not a
  general law.
- At this (deliberately tiny) training budget the static MobileNetV2 is a
  serious competitor: 92.67% accuracy at 0.99× speed vs the early-exit's
  92.92% at 1.10×. The early-exit model wins on both axes, but narrowly —
  with longer training or a different backbone the comparison could flip,
  which is why the fair runner makes the static baseline a first-class
  citizen.
- These numbers characterize *this* bounded run on *this* laptop GPU. They
  are not Jetson numbers, not ImageNet numbers, and not a general speedup
  claim.
