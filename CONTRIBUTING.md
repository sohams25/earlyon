# contributing to earlyon

## quick start

```bash
git clone https://github.com/sohams-web/earlyon.git
cd earlyon
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## what's in scope for v0.1

- ResNet18, ResNet50, MobileNetV2 wrappers
- two-stage trainer (backbone first, then frozen-backbone + exit heads)
- confidence-based routing
- post-hoc temperature scaling
- greedy threshold calibration
- throughput / accuracy benchmarks
- Jetson power/thermal profiler

## explicitly out of scope until later

- batched per-sample routing (v0.1 is batch=1)
- ONNX export with conditional control flow
- joint training of backbone + exits
- entropy or budget routing policies
- EfficientNet / RegNet wrappers
- TensorRT integration
- distillation between exits

## conventions

- new exit-attached architectures go in `earlyon/models/`
- tests first; tiny fixtures live in `tests/fixtures/`
- GPU-required tests get `@pytest.mark.gpu` so CI skips them
- format with `black` + `isort`, lint with `ruff`
- commit messages: lowercase, present tense, no body unless needed

## bugs / proposals

issues with a minimal repro are best. for new architectures, please include
the exit-point placement rationale (which layers, why those, default thresholds).
