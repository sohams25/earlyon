"""Bounded, reproducible CUDA evidence run for release readiness.

One seeded experiment on CIFAR-10 (public, auto-downloaded) intended to
*validate the system end-to-end*, not to maximize a headline number:

- model: ResNet-18 early-exit wrapper, fine-tuned from ImageNet-pretrained
  weights for a small, explicit number of epochs;
- static smaller baseline: MobileNetV2 (ImageNet-pretrained), fine-tuned the
  same way — a legitimately smaller static model trained on the same data;
- explicit disjoint splits: train (45k), temperature (2.5k), calibration
  (2.5k), official test (10k). The threshold search never sees test labels.
- fair benchmark: all three models measured by benchmark_models on the exact
  same test samples with identical boundaries; a noise-input variant is
  recorded separately as a best-case bound.

Hard bounds: explicit epoch caps, a wall-clock budget check, one fixed seed.
Note: full bitwise determinism is not guaranteed on GPU (cuDNN autotuning);
the seed fixes data order and initialisation.

Output: docs/evidence/cuda_evidence.json + docs/evidence/CUDA_EVIDENCE.md.
Negative results (e.g. wrapper slower than backbone) are recorded as-is.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from earlyon import __version__  # noqa: E402
from earlyon.benchmarking import benchmark_models, evaluate  # noqa: E402
from earlyon.core.thresholds import calibrate_thresholds  # noqa: E402
from earlyon.training import stage1_train_backbone, stage2_train_exits  # noqa: E402
from earlyon.utils import build_model, save_wrapper  # noqa: E402

SEED = 42
WALL_BUDGET_S = 3 * 3600
OUT_DIR = ROOT / "docs" / "evidence"

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def loaders(root: str, image_size: int = 224, batch_size: int = 128):
    train_tf = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    train_full = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    eval_train = torchvision.datasets.CIFAR10(root, train=True, download=False, transform=eval_tf)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=eval_tf)

    # explicit disjoint splits over the 50k train images
    n = len(train_full)
    temp_idx = list(range(0, 2500))
    calib_idx = list(range(2500, 5000))
    train_idx = list(range(5000, n))

    mk = lambda ds, idx, bs, shuffle: DataLoader(  # noqa: E731
        Subset(ds, idx), batch_size=bs, shuffle=shuffle, num_workers=2, pin_memory=True
    )
    return {
        "train": mk(train_full, train_idx, batch_size, True),
        "temperature": mk(eval_train, temp_idx, batch_size, False),
        "calibration": mk(eval_train, calib_idx, batch_size, False),
        "test_batched": DataLoader(test, batch_size=batch_size, num_workers=2),
        "test_b1": DataLoader(test, batch_size=1, num_workers=2, pin_memory=True),
        "counts": {
            "train": len(train_idx),
            "temperature": len(temp_idx),
            "calibration": len(calib_idx),
            "test": len(test),
        },
    }


def batched_accuracy(module: torch.nn.Module, loader: DataLoader, device: str) -> float:
    module = module.to(device).eval()
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            pred = module(x.to(device)).argmax(dim=-1).cpu()
            correct += int((pred == y).sum())
            total += y.numel()
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data"))
    parser.add_argument("--epochs-stage1", type=int, default=2)
    parser.add_argument("--epochs-stage2", type=int, default=2)
    parser.add_argument("--epochs-baseline", type=int, default=2)
    parser.add_argument("--bench-runs", type=int, default=300)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA is required for the evidence run")
    device = "cuda"
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    t_start = time.time()

    def budget_check(stage: str) -> None:
        spent = time.time() - t_start
        if spent > WALL_BUDGET_S:
            sys.exit(f"wall-clock budget exceeded before {stage} ({spent:.0f}s)")

    data = loaders(args.data)
    log = {"stages": {}}

    # ---- early-exit ResNet-18 ----
    model = build_model("resnet18", num_classes=10, pretrained=True)
    t0 = time.time()
    stage1_train_backbone(
        model.backbone, data["train"], epochs=args.epochs_stage1, lr=0.005, device=device
    )
    log["stages"]["stage1_seconds"] = round(time.time() - t0, 1)
    budget_check("stage2")
    t0 = time.time()
    stage2_train_exits(model, data["train"], epochs=args.epochs_stage2, lr=1e-3, device=device)
    log["stages"]["stage2_seconds"] = round(time.time() - t0, 1)
    budget_check("baseline")

    # ---- static smaller baseline: MobileNetV2 backbone ----
    baseline = build_model("mobilenetv2", num_classes=10, pretrained=True)
    t0 = time.time()
    stage1_train_backbone(
        baseline.backbone, data["train"], epochs=args.epochs_baseline, lr=0.002, device=device
    )
    log["stages"]["baseline_seconds"] = round(time.time() - t0, 1)
    budget_check("calibration")

    # ---- calibration: temperatures on the temperature split, thresholds on
    # the calibration split; the test set is never touched here ----
    t0 = time.time()
    calib = calibrate_thresholds(
        model,
        data["calibration"],
        target_accuracy_drop=0.01,
        device=device,
        fit_temperature=True,
        temperature_loader=data["temperature"],
    )
    log["stages"]["calibrate_seconds"] = round(time.time() - t0, 1)
    budget_check("evaluation")

    # ---- held-out test evaluation ----
    t0 = time.time()
    report = evaluate(model, data["test_b1"], device=device)
    log["stages"]["test_eval_seconds"] = round(time.time() - t0, 1)
    backbone_test_acc = batched_accuracy(model.backbone, data["test_batched"], device)
    baseline_test_acc = batched_accuracy(baseline.backbone, data["test_batched"], device)
    budget_check("benchmark")

    # ---- fair benchmark: identical real test samples for all three models ----
    torch.cuda.reset_peak_memory_stats()
    cmp_real = benchmark_models(
        {
            "backbone_resnet18": model.backbone,
            "early_exit_resnet18": model,
            "static_mobilenetv2": baseline.backbone,
        },
        loader=data["test_b1"],
        device=device,
        num_warmup=50,
        num_runs=args.bench_runs,
    )
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6
    cmp_noise = benchmark_models(
        {
            "backbone_resnet18": model.backbone,
            "early_exit_resnet18": model,
            "static_mobilenetv2": baseline.backbone,
        },
        input_shape=(1, 3, 224, 224),
        device=device,
        num_warmup=50,
        num_runs=args.bench_runs,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_wrapper(model, OUT_DIR / "resnet18_ee_evidence.pth")

    result = {
        "purpose": "bounded system-validation run; not a tuned headline benchmark",
        "seed": SEED,
        "determinism_note": (
            "seed fixes init/data order; bitwise GPU determinism not enforced (cuDNN)"
        ),
        "dataset": {"name": "CIFAR-10 (torchvision, images upsampled to 224)", **data["counts"]},
        "weights": "ImageNet-pretrained torchvision weights, fine-tuned as recorded below",
        "epochs": {
            "stage1_backbone": args.epochs_stage1,
            "stage2_exits": args.epochs_stage2,
            "static_baseline": args.epochs_baseline,
        },
        "model": {
            "backbone": "resnet18",
            "exit_points": [
                {"name": ep.name, "layer": ep.layer_name, "in_channels": ep.in_channels}
                for ep in model.config.exit_points
            ],
            "head_config": "EarlyExitHead(hidden_dim=128, dropout=0.2)",
        },
        "calibration": {
            "objective": "accuracy_budget",
            "target_accuracy_drop": 0.01,
            "method": calib.method,
            "thresholds": calib.thresholds,
            "enabled_exits": calib.enabled_exits,
            "temperatures": calib.temperatures,
            "temperature_fit_converged": (
                {k: v.converged for k, v in calib.temperature_fits.items()}
                if calib.temperature_fits
                else None
            ),
            "calibration_split_baseline_acc": round(calib.baseline_accuracy, 4),
            "calibration_split_routed_acc": round(calib.final_accuracy, 4),
        },
        "test": {
            "early_exit_routed_accuracy": round(report.overall_accuracy, 4),
            "backbone_accuracy": round(backbone_test_acc, 4),
            "static_mobilenetv2_accuracy": round(baseline_test_acc, 4),
            "early_exit_estimated_flops_fraction": round(report.avg_computation_used, 4),
            "exit_distribution": report.exit_distribution,
        },
        "benchmark_real_input": {name: r.to_dict() for name, r in cmp_real.results.items()},
        "benchmark_real_speedups": {
            "early_exit_vs_backbone": round(
                cmp_real.speedup_vs("early_exit_resnet18", "backbone_resnet18"), 3
            ),
            "static_vs_backbone": round(
                cmp_real.speedup_vs("static_mobilenetv2", "backbone_resnet18"), 3
            ),
        },
        "benchmark_noise_input": {
            "note": "best-case bound: trained heads may fire spuriously on noise",
            **{name: r.to_dict() for name, r in cmp_noise.results.items()},
            "early_exit_vs_backbone_speedup": round(
                cmp_noise.speedup_vs("early_exit_resnet18", "backbone_resnet18"), 3
            ),
        },
        "peak_cuda_memory_mb_during_real_benchmark": round(peak_mem_mb, 1),
        "stage_seconds": log["stages"],
        "total_wall_seconds": round(time.time() - t_start, 1),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "earlyon": __version__,
            "os": platform.platform(),
        },
    }
    (OUT_DIR / "cuda_evidence.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_DIR / 'cuda_evidence.json'}")


if __name__ == "__main__":
    main()
