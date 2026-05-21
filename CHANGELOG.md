# changelog

format follows [keep a changelog](https://keepachangelog.com/en/1.1.0/).

## unreleased

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
