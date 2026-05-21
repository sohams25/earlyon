# CLAUDE.md — earlyon Workspace

> **Project:** Production-Ready Early Exit for PyTorch CV Models  
> **Domain:** Neural network optimization / efficient inference for edge AI  
> **Language:** Python 3.10+  
> **Key External Dependencies:** PyTorch, torchvision, ONNX  
> **Hardware Required:** GPU for training; NVIDIA Jetson for README benchmarks

---

## 1. Overview & Engineering Philosophy

**earlyon** is a pip-installable Python library that wraps standard PyTorch CV models (ResNet, MobileNet, EfficientNet) with early exit points — lightweight classifiers attached to intermediate layers that allow "easy" inputs to exit the network early, saving compute. Benchmarked on NVIDIA Jetson with real throughput numbers.

**The product thesis:** 8+ years of research prove early exits work (1.3–2.5× speedup), but no production toolkit exists. Every paper provides custom code for their specific architecture. earlyon is the first tool any practitioner can `pip install` and use.

**Engineering principles:**
- **Research must be made practical.** Papers show speedups; earlyon ships a toolkit with hardware benchmarks.
- **Training must be simple.** Two-stage training: train backbone normally, then freeze and train exit heads. No modification to standard training pipelines.
- **Hardware benchmarks are non-negotiable.** Without Jetson FPS numbers, this is just another PyTorch experiment. The benchmark table is the README centerpiece.
- **Exit routing must be transparent.** Every inference reports: which exit was taken, how much compute was used, and the confidence level.

---

## 2. Workspace Structure

```
earlyon/
├── earlyon/                     # Source package
│   ├── __init__.py              # Version, exports
│   ├── cli.py                   # Click CLI
│   ├── core/                    # Model wrappers, exit heads, routing
│   │   ├── __init__.py
│   │   ├── wrappers.py          # EarlyExitWrapper base class
│   │   ├── exit_head.py         # Lightweight classifier head
│   │   ├── router.py            # Confidence/entropy/budget routing
│   │   └── thresholds.py        # Threshold calibration
│   ├── training/                # Training loops, losses
│   │   ├── __init__.py
│   │   ├── joint_trainer.py     # Joint: train backbone + exits together
│   │   ├── two_stage_trainer.py # Two-stage: backbone first, exits second
│   │   └── losses.py            # Weighted multi-exit cross-entropy
│   ├── benchmarking/            # Throughput, latency, power
│   │   ├── __init__.py
│   │   ├── throughput.py        # Images/sec measurement
│   │   ├── accuracy_vs_exit.py  # Per-exit accuracy breakdown
│   │   └── jetson_profiler.py   # tegrastats integration
│   ├── models/                  # Pre-configured architectures
│   │   ├── __init__.py
│   │   ├── resnet50_ee.py       # ResNet50 + 3 exit points
│   │   ├── resnet18_ee.py       # ResNet18 + 2 exit points
│   │   ├── mobilenetv2_ee.py    # MobileNetV2 + 2 exit points
│   │   └── custom.py            # Wrap any user-provided model
│   └── utils.py                 # Shared helpers
├── tests/
│   ├── test_wrappers.py         # Forward pass (training + inference modes)
│   ├── test_training.py         # Loss computation, convergence
│   ├── test_router.py           # Routing policies
│   ├── test_thresholds.py       # Calibration accuracy
│   ├── test_benchmarks.py       # Statistical correctness
│   └── fixtures/                # Tiny models for fast testing
├── examples/
│   ├── 01_train_resnet50_cifar10.py
│   ├── 02_benchmark_throughput.py
│   ├── 03_jetson_deployment.py
│   └── 04_custom_model.py
├── setup.py
├── pyproject.toml
├── .github/workflows/ci.yml
└── README.md                    # Benchmark table is the centerpiece
```

**Rule:** `core/` contains model logic only. `training/` contains training logic only. Neither imports from the other at module level. The CLI and user code compose them.

---

## 3. Everything-Claude-Code (ECC) Orchestration

### 3.1 Active Subagent Stack

| Subagent | Role | When to Invoke |
|----------|------|----------------|
| **TDD-Agent** | Writes test specs before implementation | Every new wrapper, trainer, router, benchmark module |
| **Model-Surgeon** | Designs exit point placement for new architectures | When adding support for a new backbone (EfficientNet, RegNet, etc.) |
| **Trainer-Designer** | Designs training loops and loss functions | When modifying training strategy or adding a new trainer |
| **Benchmark-Verifier** | Validates hardware measurement methodology | Before committing any FPS/latency/power numbers |

### 3.2 Pre-Execution Hook (PreToolUse)

Before generating any code, evaluate:

1. **Is this a new model wrapper or a modification?** If new → invoke Model-Surgeon for exit placement, then TDD-Agent for tests.
2. **Does this code require GPU for testing?** Mark GPU-dependent tests with `@pytest.mark.gpu`. CI runs CPU-only tests.
3. **Does the change affect training dynamics?** If yes, invoke Trainer-Designer to review loss computation and gradient flow.
4. **Will this code be benchmarked on Jetson?** If yes, ensure the implementation is deterministic (no randomness in inference path).

### 3.3 Post-Execution Hook

After any file modification:

```bash
black <file> && isort <file>
python -m py_compile <file>
```

After wrapper/core modification:
```bash
pytest tests/test_wrappers.py -v --tb=short
```

After training module modification:
```bash
pytest tests/test_training.py -v --tb=short
# If GPU available:
pytest tests/test_training.py -v --tb=short -m gpu
```

After benchmark module modification:
```bash
pytest tests/test_benchmarks.py -v --tb=short
```

### 3.4 State Store Policy

Append training insights and hardware quirks to `.claude_state.jsonl`:

```json
{"timestamp": "2026-05-21T14:00:00Z", "project": "earlyon", "category": "training", "lesson": "Two-stage training converges 3× faster than joint training for exit heads. Joint training causes gradient conflict between backbone and exits.", "affects": "training/two_stage_trainer.py", "source": "ACM 2024 survey Section 5.4"}
{"timestamp": "2026-05-21T14:30:00Z", "project": "earlyon", "category": "hardware", "lesson": "Jetson Orin NX in MAXN mode: must run 50-iteration warmup before timing. GPU clocks don't stabilize until ~30 iterations.", "affects": "benchmarking/jetson_profiler.py"}
```

---

## 4. Development Commands

### Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[torch,dev]"
```

### Run Test Suite (CPU only)

```bash
pytest tests/ -v --cov=earlyon --cov-report=term-missing -m "not gpu"
```

### Run GPU-Dependent Tests

```bash
pytest tests/ -v -m gpu
```

### Single Test

```bash
pytest tests/test_wrappers.py::test_inference_routes_to_early_exit -v
```

### Type Check

```bash
mypy earlyon/ --strict
```

### Lint & Format

```bash
black . && isort .
```

### Run Examples

```bash
python examples/01_train_resnet50_cifar10.py
python examples/02_benchmark_throughput.py
```

### Install in Editable Mode

```bash
pip install -e ".[torch,dev]"
```

---

## 5. Architecture & TDD Mandates

### 5.1 Test-Driven Development — Strict

**Every wrapper, trainer, router, and benchmark module must have tests before implementation.**

The TDD cycle:
1. TDD-Agent writes the test.
2. Run the test. Confirm it fails.
3. Implement the minimal code.
4. Refactor if needed.
5. Commit: `test: add X` → `feat: implement X`.

**Example — Adding a new model wrapper:**

```python
# Step 1: TDD-Agent writes this FIRST (tests/test_wrappers.py)
def test_resnet50_inference_returns_inference_result():
    """Forward pass in inference mode must return InferenceResult, not raw tensor."""
    model = create_resnet50_ee(num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    result = model(x, mode="inference")
    
    assert isinstance(result, InferenceResult)
    assert result.prediction.shape == (1, 10)
    assert 0.0 <= result.computation_used <= 1.0
    assert 0.0 <= result.confidence <= 1.0

def test_easy_input_exits_early():
    """Low confidence thresholds + easy input = early exit."""
    model = create_resnet50_ee(num_classes=10, thresholds=[0.01, 0.01, 0.01])
    x = torch.randn(1, 3, 224, 224)
    result = model(x, mode="inference")
    
    assert result.exit_taken == 0  # Exited at first exit
    assert result.computation_used < 0.5  # Used < half the network

def test_hard_input_reaches_final():
    """Very high thresholds = no early exit, reaches final classifier."""
    model = create_resnet50_ee(num_classes=10, thresholds=[0.99, 0.99, 0.99])
    x = torch.randn(1, 3, 224, 224)
    result = model(x, mode="inference")
    
    assert result.exit_taken == -1  # No exit triggered
    assert result.computation_used == 1.0  # Full network
```

```python
# Step 2: Implement the wrapper SECOND (earlyon/core/wrappers.py)
class EarlyExitWrapper(nn.Module):
    def forward(self, x, mode="inference"):
        if mode == "training":
            return self._forward_training(x)
        return self._forward_inference(x)
    
    def _forward_inference(self, x) -> InferenceResult:
        # ... minimal code to pass the three tests above
```

### 5.2 Wrapper Architecture

The `EarlyExitWrapper` base class is the project's heart. It must:

1. **Register exit heads** at specified backbone layers.
2. **Support two forward modes:** `training` (all exits) and `inference` (route to first confident exit).
3. **Report computation used** as a fraction of total layers.
4. **Expose exit parameters separately** for two-stage training (freeze backbone, train exits).

```python
class EarlyExitWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, exit_configs: List[ExitConfig]):
        super().__init__()
        self.backbone = backbone
        self.exit_heads = nn.ModuleDict()
        self._register_exits(exit_configs)
    
    def _register_exits(self, configs: List[ExitConfig]):
        """Attach exit heads to specified backbone layers."""
        for cfg in configs:
            # Hook after the specified layer
            self.exit_heads[cfg.layer_name] = EarlyExitHead(
                in_channels=cfg.feature_channels,
                num_classes=cfg.num_classes
            )
    
    def exit_parameters(self) -> Iterator[nn.Parameter]:
        """Return only exit head parameters (for two-stage training)."""
        for head in self.exit_heads.values():
            yield from head.parameters()
    
    def _forward_training(self, x) -> List[torch.Tensor]:
        """All exits produce predictions. Used for training."""
        ...
    
    def _forward_inference(self, x) -> InferenceResult:
        """Route to earliest confident exit. Used for deployment."""
        ...
```

### 5.3 Exit Head Design

The exit head is intentionally lightweight — it's a quick classifier, not a replacement for the full model.

```python
class EarlyExitHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.classifier(x)
```

**Typical sizes:**
- 512 channels → 10 classes: ~5K parameters
- 1024 channels → 100 classes: ~13K parameters
- A full ResNet50 FC layer: 2M+ parameters

### 5.4 Two-Stage Training (Default Strategy)

Research (ACM 2024 survey) shows two-stage training is the most practical:

```python
class TwoStageTrainer:
    def stage1_train_backbone(self, train_loader, val_loader, epochs=90, lr=0.1):
        """Train backbone as standard classifier. Exits are NOT used."""
        # Standard training loop — identical to training a normal ResNet
        optimizer = torch.optim.SGD(self.model.backbone.parameters(), lr=lr)
        for epoch in range(epochs):
            for images, labels in train_loader:
                pred = self.model.backbone(images)  # Use backbone directly
                loss = F.cross_entropy(pred, labels)
                ...
    
    def stage2_train_exits(self, train_loader, val_loader, epochs=20, lr=0.001):
        """Freeze backbone. Train only exit heads."""
        # Freeze ALL backbone parameters
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        
        optimizer = torch.optim.Adam(self.model.exit_parameters(), lr=lr)
        for epoch in range(epochs):
            for images, labels in train_loader:
                all_preds = self.model(images, mode="training")
                loss = weighted_multi_exit_loss(all_preds, labels, self.loss_weights)
                ...
```

**Why two-stage over joint?**
- No modification to backbone training (use existing recipes)
- Exit training is fast (small heads, frozen backbone)
- No gradient conflict between exits
- Easy to add exits to already-trained models

### 5.5 Threshold Calibration

Thresholds control the accuracy-speed tradeoff. Calibrate on a validation set:

```python
def calibrate_thresholds(
    model: EarlyExitWrapper,
    val_loader: DataLoader,
    target_accuracy_drop: float = 0.01
) -> List[float]:
    """
    Grid search to find thresholds that keep accuracy within target drop.
    Strategy: start conservative (high thresholds), gradually lower until
    accuracy drop exceeds target.
    """
    baseline_acc = evaluate_full_model(model, val_loader)
    
    best = [1.0] * len(model.exit_points)
    for exit_idx in range(len(model.exit_points)):
        for threshold in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.5]:
            test_thresholds = best.copy()
            test_thresholds[exit_idx] = threshold
            acc, speedup = evaluate_with_thresholds(model, val_loader, test_thresholds)
            
            if baseline_acc - acc <= target_accuracy_drop:
                best[exit_idx] = threshold
            else:
                break
    return best
```

---

## 6. Hardware Benchmarking (Non-Negotiable)

### 6.1 Benchmark Targets

**Laptop benchmarks (CI-friendly, CPU + CUDA):**

| Model | Dataset | Device | Baseline FPS | With earlyon FPS | Speedup | Accuracy |
|-------|---------|--------|-------------|-----------------|---------|----------|
| ResNet50 | CIFAR-10 | RTX 3060 | 312 | 687 | **2.2×** | 94.2% (94.5%) |
| MobileNetV2 | CIFAR-10 | RTX 3060 | 892 | 1245 | **1.4×** | 93.8% (93.9%) |

**Jetson benchmarks (for README — run once manually):**

| Model | Dataset | Device | Baseline FPS | With earlyon FPS | Speedup | Avg FLOPs Used |
|-------|---------|--------|-------------|-----------------|---------|----------------|
| ResNet50 | CIFAR-10 | Jetson Orin NX | 45 | 98 | **2.2×** | 48% |
| MobileNetV2 | CIFAR-100 | Jetson Orin NX | 38 | 58 | **1.5×** | 58% |

*Accuracy in parentheses = full model (no early exit) baseline.*

### 6.2 Measurement Protocol

```python
def benchmark_throughput(model, input_shape, device, num_warmup=50, num_runs=500):
    """Strict measurement protocol for reproducible benchmarks."""
    model = model.to(device).eval()
    dummy = torch.randn(input_shape, device=device)
    
    # 1. Warmup: GPU clocks stabilize, caches warm
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy)
    
    # 2. Synchronize before timing
    if device == "cuda":
        torch.cuda.synchronize()
    
    # 3. Measure
    import time, statistics
    latencies = []
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(num_runs):
            iter_start = time.perf_counter()
            _ = model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - iter_start)
        total = time.perf_counter() - start
    
    return {
        "throughput_ips": num_runs / total,
        "latency_median_ms": statistics.median(latencies) * 1000,
        "latency_p95_ms": sorted(latencies)[int(num_runs * 0.95)] * 1000,
    }
```

### 6.3 Jetson Profiler

```python
class JetsonProfiler:
    """Profile inference on Jetson with power and temperature monitoring."""
    
    def profile(self, model, input_shape, num_runs=100):
        results = []
        for _ in range(num_runs):
            stats_before = self._read_tegrastats()
            
            start = time.perf_counter()
            with torch.no_grad():
                result = model(torch.randn(input_shape).cuda(), mode="inference")
            latency = time.perf_counter() - start
            
            stats_after = self._read_tegrastats()
            
            results.append({
                "latency_ms": latency * 1000,
                "exit_taken": result.exit_taken,
                "gpu_util": stats_after.get("GR3D", 0),
                "temp_c": stats_after.get("PLL@", 0),
                "power_mw": stats_after.get("POM_5V_IN", 0),
            })
        return results
```

### 6.4 Exit Distribution Analysis

Report what fraction of inputs exit at each point. This is a compelling interview detail:

```python
def analyze_exit_distribution(model, dataloader):
    """Return exit counts per class — reveals what the model finds easy vs hard."""
    exit_counts = defaultdict(lambda: [0, 0, 0, 0])  # exit_0, exit_1, exit_2, final
    
    for images, labels in dataloader:
        for img, label in zip(images, labels):
            result = model(img.unsqueeze(0), mode="inference")
            class_name = CLASS_NAMES[label.item()]
            exit_idx = result.exit_taken if result.exit_taken >= 0 else 3
            exit_counts[class_name][exit_idx] += 1
    
    return exit_counts
```

**Example output:**
```
automobile:  [62%, 25%, 10%, 3%]  → exits early (distinctive)
bird:        [15%, 35%, 30%, 20%] → goes deep (ambiguous)
cat:         [18%, 32%, 28%, 22%] → goes deep (ambiguous)
```

---

## 7. CLI Design

```bash
# Wrap a pretrained model with early exits
earlyon create --backbone resnet50 --num-classes 10 --exits 3 --output model.py

# Two-stage training
earlyon train backbone --model model.py --dataset cifar10 --epochs 90
earlyon train exits --model model.py --dataset cifar10 --epochs 20

# Calibrate thresholds
earlyon calibrate --model checkpoint.pth --dataset cifar10 --target-drop 0.01

# Benchmark
earlyon benchmark --model checkpoint.pth --dataset cifar10 --device cuda

# Jetson profile
earlyon profile --model checkpoint.pth --dataset cifar10 --device jetson

# Export to ONNX
earlyon export --model checkpoint.pth --output model.onnx --dynamic-batch

# Exit distribution analysis
earlyon analyze exits --model checkpoint.pth --dataset cifar10
```

---

## 8. ECC Subagent Prompts

### TDD-Agent

```
You are TDD-Agent for earlyon. Write pytest test files before any implementation exists.

For the module described below, produce a complete test file with:
1. Happy path: typical correct forward pass
2. Boundary: empty batch, single sample, maximum number of exits
3. Failure mode: mismatched dimensions, missing exit config, invalid routing policy
4. Shape invariants: output shapes match expectations regardless of exit point

Use tiny models (2-layer CNN) for fast tests. Do not depend on torchvision downloads in tests.
Mark GPU-dependent tests with @pytest.mark.gpu.
Output: tests/test_{module_name}.py
```

### Model-Surgeon

```
You are Model-Surgeon. Design exit point placement for a new backbone architecture.

Given: {BACKBONE_NAME} with layer structure: {LAYER_LIST}
Decide:
1. How many exit points? (typically 2-4)
2. Where to place them? (after which layers)
3. What are the feature channel dimensions at each exit point?
4. What are reasonable default confidence thresholds?

Principles:
- Early exits should capture low-level features (edges, textures)
- Later exits capture high-level features (shapes, objects)
- Channel dimensions should grow with depth
- Computation_used at each exit should be roughly evenly spaced

Output: Exit placement config + rationale.
```

### Benchmark-Verifier

```
You are Benchmark-Verifier. Validate that the benchmarking methodology is statistically sound.

Check:
1. Is warmup sufficient (≥50 iterations) for GPU clock stabilization?
2. Is CUDA synchronized before timing measurements?
3. Is sample size ≥500 for statistical significance?
4. Are we reporting median (not mean) for latency?
5. Is the comparison fair (same batch size, precision, power mode)?
6. Are environment details documented (JetPack, CUDA, PyTorch versions)?

Benchmark code: {CODE}
Output: Verification report with specific fixes.
```

---

## 9. Quick Reference

| Situation | Action |
|-----------|--------|
| Adding a new backbone (e.g., EfficientNet) | Invoke Model-Surgeon → TDD-Agent → implement → test |
| Training loss not converging | Check: learning rate, loss weights sum to ~1.0, exit head initialization |
| No speedup in benchmarks | Check: thresholds too high (no exits triggered), batch size mismatch, no CUDA sync |
| Accuracy drop too large | Run threshold calibration with stricter target_drop (0.005 instead of 0.01) |
| Jetson benchmark numbers look wrong | Invoke Benchmark-Verifier. Check power mode, warmup, CUDA sync |
| Test passes on GPU but fails on CPU | Model behavior may differ slightly — mark test with `@pytest.mark.gpu` |
| Ready to release | Full test suite → Jetson benchmarks → README with table → demo GIF → PyPI |