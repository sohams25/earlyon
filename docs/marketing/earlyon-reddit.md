# r/MachineLearning launch draft

Flair: [P] (Project). Post as text, not a link post; r/ML buries bare links.
Best window: Tue–Thu, 14:00–16:00 UTC.

---

## Title

[P] earlyon: skip the last ResNet layers when the network is already
confident. On CIFAR-10, 58% of images never reach layer4.

## Body

A deep network runs every layer for every image, even the obvious ones. A
frontal close-up of a car and a blurry half-occluded bird both pay the full
forward pass.

Early-exit networks fix this: attach small classifier heads partway through
the backbone, and let confident predictions leave early. The research is
mature. BranchyNet was 2016, and a 2024 ACM survey covers eight years of
follow-ups showing 1.3–2.5× compute savings at single-sample edge inference.
But every paper ships custom code for its one architecture. There was no tool
you could `pip install`, so I built one:

```
pip install earlyon
```

```python
from earlyon.models import resnet50_ee

model = resnet50_ee(num_classes=10, pretrained=True).eval()
result = model(x, mode="inference")

result.exit_taken        # which head fired (-1 = full network)
result.computation_used  # fraction of FLOPs actually run
result.confidence        # how sure it was when it left
```

Numbers (CIFAR-10, trained on a laptop RTX 4050, thresholds calibrated to a
1% accuracy budget):

| Model | Test Acc | Baseline | Avg FLOPs used | % exited early |
|---|---:|---:|---:|---:|
| ResNet18 | 94.42% | 96.32% | 89.88% | 35.3% |
| ResNet50 | 95.88% | 97.60% | 81.42% | 58.2% |
| MobileNetV2 | 93.31% | 95.08% | 93.90% | 8.5% |

The headline stat: with thresholds calibrated for a ~1% accuracy budget,
58% of CIFAR-10 images exit ResNet50 before layer4, cutting ~19% of FLOPs on
the real test set. I'm only publishing FLOPs measured on real test images.
Wall-clock depends on your hardware and input mix, and I don't have a Jetson
number I trust enough to print yet. The repo ships a Jetson profiler (power +
thermals via `tegrastats`) so you can measure yours.

Things I tried to get right:

- The tool reports average FLOPs actually skipped on real test images, not
  the theoretical best case. It also tells you when early exit doesn't help.
  I left the MobileNetV2 row in even though it loses: that backbone is
  already cheap, so only 8.5% of images exit early.
- No backbone surgery. Exit heads attach via forward hooks, so your existing
  weights load unchanged. `custom_ee` wraps any `nn.Module`, including ViT
  (token-pooled exits after encoder blocks).
- Training stays your training. Two-stage by default: train the backbone with
  your existing recipe (or start from any pretrained checkpoint), then freeze
  it and train the tiny heads for a few epochs. Joint training ships too if
  you want peak accuracy.
- Thresholds come from a stated budget, not hand-tuning. Give an accuracy
  budget ("within 1% of baseline") or a compute budget ("80% of FLOPs on
  average") and a greedy sweep finds per-exit thresholds, with optional
  temperature scaling first because raw softmax confidence is miscalibrated.

Limitations, since r/ML will ask: per-sample routing is batch-1 (there's a
batched mode where the whole batch exits together); `torch.compile` can't
trace the routing control flow (compile the raw backbone instead); ONNX
export is static-graph only, all exits computed and routing left to the
caller. Results above are CIFAR-10; ImageNet-scale benchmarks are the next
milestone.

Code, benchmarks, and the design-decisions doc:
https://github.com/sohams25/earlyon

Happy to answer questions about the calibration method, the hook + sentinel
routing mechanism, or edge deployment.

---

## First comment (post immediately, from the same account)

Design decisions write-up (why two-stage training, why greedy calibration, why
temperature scaling, why entropy routing ships alongside confidence):
https://github.com/sohams25/earlyon/blob/main/docs/DESIGN_DECISIONS.md

Raw benchmark JSON, reproducible with `scripts/run_benchmarks.py`:
https://github.com/sohams25/earlyon/blob/main/docs/benchmarks.json
