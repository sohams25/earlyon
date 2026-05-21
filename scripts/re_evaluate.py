"""Re-evaluate already-trained checkpoints with fixed FLOPs accounting.

Skips training; just runs test-eval + throughput bench and rewrites
docs/benchmarks.json with the corrected avg_computation_used.
"""

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from earlyon.benchmarking import benchmark_backbone, benchmark_wrapper, evaluate
from earlyon.utils import cifar10_loaders, load_wrapper

OUT_JSON = ROOT / "docs" / "benchmarks.json"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}

    _, _, test_loader = cifar10_loaders(batch_size=1, num_workers=2)

    for backbone in ["resnet18", "resnet50", "mobilenetv2"]:
        ckpt = ROOT / "docs" / f"{backbone}_cifar10.pth"
        if not ckpt.exists():
            print(f"[{backbone}] no checkpoint at {ckpt}, skipping")
            continue
        print(f"[{backbone}] loading {ckpt}")
        model = load_wrapper(ckpt)
        print(f"[{backbone}] _flops_at = {model._flops_at}")

        # test-set evaluation with FIXED flops
        print(f"[{backbone}] test eval", flush=True)
        report = evaluate(model, test_loader, device=device)

        # throughput bench
        print(f"[{backbone}] throughput", flush=True)
        shape = (1, 3, 224, 224)
        wrap_r = benchmark_wrapper(
            model, input_shape=shape, device=device, num_warmup=50, num_runs=300,
        )
        bb_r = benchmark_backbone(
            model.backbone, input_shape=shape, device=device, num_warmup=50, num_runs=300,
        )
        speedup = wrap_r.throughput_ips / bb_r.throughput_ips

        # merge into existing record
        existing = results.get(backbone, {})
        existing.update({
            "backbone": backbone,
            "dataset": "cifar10",
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "test_accuracy": round(report.overall_accuracy, 4),
            "test_avg_computation_used": round(report.avg_computation_used, 4),
            "test_exit_distribution": report.exit_distribution,
            "throughput_backbone_ips": round(bb_r.throughput_ips, 1),
            "throughput_wrapper_ips_noise_input": round(wrap_r.throughput_ips, 1),
            "throughput_speedup_noise_input": round(speedup, 3),
            "latency_backbone_p50_ms": round(bb_r.latency_p50_ms, 3),
            "latency_wrapper_p50_ms_noise_input": round(wrap_r.latency_p50_ms, 3),
            "note": "throughput uses random noise input which may trigger spurious exits in trained heads; test_avg_computation_used is the honest signal",
        })
        results[backbone] = existing
        OUT_JSON.write_text(json.dumps(results, indent=2))
        print(f"[{backbone}] acc={report.overall_accuracy:.4f} "
              f"avg_comp={report.avg_computation_used:.4f} "
              f"dist={report.exit_distribution}", flush=True)

    print("all done")


if __name__ == "__main__":
    main()
