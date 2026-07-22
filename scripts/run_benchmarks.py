"""End-to-end: train, calibrate, benchmark each backbone on CIFAR-10.

Writes a JSON record per backbone to docs/benchmarks.json so the README can
be populated from real numbers. Designed to run unattended on a single GPU.

Usage:
    python scripts/run_benchmarks.py [resnet18|resnet50|mobilenetv2 ...]
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from earlyon.benchmarking import benchmark_models, evaluate  # noqa: E402
from earlyon.core.thresholds import calibrate_thresholds  # noqa: E402
from earlyon.training import stage1_train_backbone, stage2_train_exits  # noqa: E402
from earlyon.utils import build_model, cifar10_loaders, save_wrapper  # noqa: E402

METHODOLOGY = "v0.3-fair-runner"

OUT_DIR = ROOT / "docs"
OUT_DIR.mkdir(exist_ok=True)
OUT_JSON = OUT_DIR / "benchmarks.json"

# fine-tuning from imagenet pretrained -- a few epochs is enough for cifar10
PLAN = {
    "resnet18": {"batch": 128, "stage1_epochs": 4, "stage2_epochs": 4, "lr1": 0.005},
    "resnet50": {"batch": 32, "stage1_epochs": 3, "stage2_epochs": 3, "lr1": 0.005},
    "mobilenetv2": {"batch": 32, "stage1_epochs": 4, "stage2_epochs": 4, "lr1": 0.002},
}


def device_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def run_one(backbone: str) -> dict:
    cfg = PLAN[backbone]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_prefix = f"[{backbone}]"
    print(f"{log_prefix} starting on {device_name()}", flush=True)

    train_loader, val_loader, test_loader = cifar10_loaders(
        batch_size=cfg["batch"],
        num_workers=2,
    )
    model = build_model(backbone, num_classes=10, pretrained=True)

    t0 = time.time()
    print(f"{log_prefix} stage1: {cfg['stage1_epochs']} epochs", flush=True)
    stage1_train_backbone(
        model.backbone,
        train_loader,
        epochs=cfg["stage1_epochs"],
        lr=cfg["lr1"],
        device=device,
        on_epoch_end=lambda log: print(
            f"{log_prefix} s1 e{log.epoch}: loss={log.loss:.4f} acc={log.accuracy:.4f}",
            flush=True,
        ),
    )
    t_stage1 = time.time() - t0

    t0 = time.time()
    print(f"{log_prefix} stage2: {cfg['stage2_epochs']} epochs", flush=True)
    stage2_train_exits(
        model,
        train_loader,
        epochs=cfg["stage2_epochs"],
        lr=1e-3,
        device=device,
        on_epoch_end=lambda log: print(
            f"{log_prefix} s2 e{log.epoch}: loss={log.loss:.4f} mean_acc={log.accuracy:.4f} "
            f"per_exit={log.per_exit_accuracy}",
            flush=True,
        ),
    )
    t_stage2 = time.time() - t0

    print(f"{log_prefix} calibrating thresholds", flush=True)
    t0 = time.time()
    calib = calibrate_thresholds(
        model,
        val_loader,
        target_accuracy_drop=0.01,
        device=device,
    )
    t_calib = time.time() - t0
    print(
        f"{log_prefix} thresholds={calib.thresholds} "
        f"baseline_acc={calib.baseline_accuracy:.4f} "
        f"final_acc={calib.final_accuracy:.4f} "
        f"avg_comp={calib.avg_computation_used:.4f}",
        flush=True,
    )

    print(f"{log_prefix} test-set evaluation", flush=True)
    report = evaluate(model, test_loader, device=device)
    print(
        f"{log_prefix} test acc={report.overall_accuracy:.4f} "
        f"avg_comp={report.avg_computation_used:.4f} "
        f"exit_dist={report.exit_distribution}",
        flush=True,
    )

    # fair comparison: identical samples + boundaries for wrapper and backbone.
    # real-input numbers are the honest signal; noise input is a best-case bound.
    print(f"{log_prefix} fair throughput (real input)", flush=True)
    cmp_real = benchmark_models(
        {"early_exit": model, "backbone": model.backbone},
        loader=test_loader,
        device=device,
        num_warmup=50,
        num_runs=300,
    )
    print(f"{log_prefix} fair throughput (noise input)", flush=True)
    cmp_noise = benchmark_models(
        {"early_exit": model, "backbone": model.backbone},
        input_shape=(1, 3, 224, 224),
        device=device,
        num_warmup=50,
        num_runs=300,
    )

    save_wrapper(model, OUT_DIR / f"{backbone}_cifar10.pth")

    return {
        "backbone": backbone,
        "dataset": "cifar10",
        "methodology": METHODOLOGY,
        "device": device_name(),
        "stage1_seconds": round(t_stage1, 1),
        "stage2_seconds": round(t_stage2, 1),
        "calibrate_seconds": round(t_calib, 1),
        "thresholds": calib.thresholds,
        "enabled_exits": calib.enabled_exits,
        "baseline_accuracy_val": round(calib.baseline_accuracy, 4),
        "test_accuracy": round(report.overall_accuracy, 4),
        "test_avg_estimated_flops_fraction": round(report.avg_computation_used, 4),
        "test_exit_distribution": report.exit_distribution,
        "real_input": {
            "early_exit": cmp_real.results["early_exit"].to_dict(),
            "backbone": cmp_real.results["backbone"].to_dict(),
            "speedup": round(cmp_real.speedup_vs("early_exit", "backbone"), 3),
        },
        "noise_input": {
            "early_exit": cmp_noise.results["early_exit"].to_dict(),
            "backbone": cmp_noise.results["backbone"].to_dict(),
            "speedup": round(cmp_noise.speedup_vs("early_exit", "backbone"), 3),
            "note": "best-case bound: trained heads may fire spuriously on noise",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backbones", nargs="*", default=list(PLAN.keys()))
    args = parser.parse_args()

    doc = {}
    if OUT_JSON.exists():
        try:
            doc = json.loads(OUT_JSON.read_text())
        except Exception:
            doc = {}
    results = doc.setdefault("runs", {})

    for backbone in args.backbones:
        if backbone not in PLAN:
            print(f"skipping unknown backbone {backbone!r}")
            continue
        try:
            results[backbone] = run_one(backbone)
            OUT_JSON.write_text(json.dumps(doc, indent=2))
            print(f"[{backbone}] DONE; wrote {OUT_JSON}", flush=True)
        except Exception as exc:
            print(f"[{backbone}] FAILED: {exc}", flush=True)
            traceback.print_exc()
            # continue with next backbone

    print("all done")


if __name__ == "__main__":
    main()
