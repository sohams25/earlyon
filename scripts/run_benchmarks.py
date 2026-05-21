"""End-to-end: train, calibrate, benchmark each backbone on CIFAR-10.

Writes a JSON record per backbone to docs/benchmarks.json so the README can
be populated from real numbers. Designed to run unattended on a single GPU.

Usage:
    python scripts/run_benchmarks.py [resnet18|resnet50|mobilenetv2 ...]
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from earlyon.benchmarking import benchmark_backbone, benchmark_wrapper, evaluate
from earlyon.core.thresholds import calibrate_thresholds
from earlyon.training import stage1_train_backbone, stage2_train_exits
from earlyon.utils import build_model, cifar10_loaders, save_wrapper

OUT_DIR = ROOT / "docs"
OUT_DIR.mkdir(exist_ok=True)
OUT_JSON = OUT_DIR / "benchmarks.json"

PLAN = {
    "resnet18":    {"batch": 128, "stage1_epochs": 20, "stage2_epochs": 10, "lr1": 0.01},
    "resnet50":    {"batch":  32, "stage1_epochs": 15, "stage2_epochs":  8, "lr1": 0.01},
    "mobilenetv2": {"batch":  96, "stage1_epochs": 20, "stage2_epochs": 10, "lr1": 0.005},
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
        batch_size=cfg["batch"], num_workers=2,
    )
    model = build_model(backbone, num_classes=10, pretrained=True)

    t0 = time.time()
    print(f"{log_prefix} stage1: {cfg['stage1_epochs']} epochs", flush=True)
    stage1_train_backbone(
        model.backbone, train_loader,
        epochs=cfg["stage1_epochs"], lr=cfg["lr1"], device=device,
        on_epoch_end=lambda l: print(
            f"{log_prefix} s1 e{l.epoch}: loss={l.loss:.4f} acc={l.accuracy:.4f}",
            flush=True,
        ),
    )
    t_stage1 = time.time() - t0

    t0 = time.time()
    print(f"{log_prefix} stage2: {cfg['stage2_epochs']} epochs", flush=True)
    stage2_train_exits(
        model, train_loader,
        epochs=cfg["stage2_epochs"], lr=1e-3, device=device,
        on_epoch_end=lambda l: print(
            f"{log_prefix} s2 e{l.epoch}: loss={l.loss:.4f} acc={l.accuracy:.4f}",
            flush=True,
        ),
    )
    t_stage2 = time.time() - t0

    print(f"{log_prefix} calibrating thresholds", flush=True)
    t0 = time.time()
    calib = calibrate_thresholds(
        model, val_loader, target_accuracy_drop=0.01, device=device,
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

    print(f"{log_prefix} throughput (wrapper vs backbone)", flush=True)
    shape = (1, 3, 224, 224)
    wrap_r = benchmark_wrapper(
        model, input_shape=shape, device=device, num_warmup=50, num_runs=300,
    )
    bb_r = benchmark_backbone(
        model.backbone, input_shape=shape, device=device, num_warmup=50, num_runs=300,
    )
    speedup = wrap_r.throughput_ips / bb_r.throughput_ips

    save_wrapper(model, OUT_DIR / f"{backbone}_cifar10.pth")

    return {
        "backbone": backbone,
        "dataset": "cifar10",
        "device": device_name(),
        "stage1_seconds": round(t_stage1, 1),
        "stage2_seconds": round(t_stage2, 1),
        "calibrate_seconds": round(t_calib, 1),
        "thresholds": calib.thresholds,
        "baseline_accuracy_val": round(calib.baseline_accuracy, 4),
        "test_accuracy": round(report.overall_accuracy, 4),
        "test_avg_computation_used": round(report.avg_computation_used, 4),
        "test_exit_distribution": report.exit_distribution,
        "throughput_backbone_ips": round(bb_r.throughput_ips, 1),
        "throughput_wrapper_ips": round(wrap_r.throughput_ips, 1),
        "speedup": round(speedup, 3),
        "latency_backbone_p50_ms": round(bb_r.latency_p50_ms, 3),
        "latency_wrapper_p50_ms": round(wrap_r.latency_p50_ms, 3),
        "latency_wrapper_p95_ms": round(wrap_r.latency_p95_ms, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backbones", nargs="*", default=list(PLAN.keys()))
    args = parser.parse_args()

    results = {}
    if OUT_JSON.exists():
        try:
            results = json.loads(OUT_JSON.read_text())
        except Exception:
            results = {}

    for backbone in args.backbones:
        if backbone not in PLAN:
            print(f"skipping unknown backbone {backbone!r}")
            continue
        try:
            results[backbone] = run_one(backbone)
            OUT_JSON.write_text(json.dumps(results, indent=2))
            print(f"[{backbone}] DONE; wrote {OUT_JSON}", flush=True)
        except Exception as exc:
            print(f"[{backbone}] FAILED: {exc}", flush=True)
            traceback.print_exc()
            # continue with next backbone

    print("all done")


if __name__ == "__main__":
    main()
