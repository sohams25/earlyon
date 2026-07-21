"""Re-evaluate already-trained checkpoints with the v0.3 fair benchmark runner.

Skips training; runs test-set evaluation plus a fair throughput comparison
(wrapper and backbone measured on the exact same sample sequence and
boundary) and writes records under the "runs" key of docs/benchmarks.json.

Pre-v0.3 records are preserved under "legacy_v0_2" and are NOT comparable:
the old runner benchmarked the wrapper and backbone on different inputs.
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from earlyon.benchmarking import benchmark_models, evaluate  # noqa: E402
from earlyon.utils import cifar10_loaders, load_wrapper  # noqa: E402

OUT_JSON = ROOT / "docs" / "benchmarks.json"
METHODOLOGY = "v0.3-fair-runner"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    doc = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    runs = doc.setdefault("runs", {})

    _, _, test_loader = cifar10_loaders(batch_size=1, num_workers=2)

    for backbone in ["resnet18", "resnet50", "mobilenetv2"]:
        ckpt = ROOT / "docs" / f"{backbone}_cifar10.pth"
        if not ckpt.exists():
            print(f"[{backbone}] no checkpoint at {ckpt}, skipping")
            continue
        print(f"[{backbone}] loading {ckpt}")
        model = load_wrapper(ckpt)

        print(f"[{backbone}] test eval", flush=True)
        report = evaluate(model, test_loader, device=device)

        # fair comparison: identical real-input samples for wrapper AND backbone
        print(f"[{backbone}] fair throughput (real input)", flush=True)
        cmp_real = benchmark_models(
            {"early_exit": model, "backbone": model.backbone},
            loader=test_loader,
            device=device,
            num_warmup=50,
            num_runs=300,
        )
        # noise input: best-case bound, same samples for both models
        print(f"[{backbone}] fair throughput (noise input)", flush=True)
        cmp_noise = benchmark_models(
            {"early_exit": model, "backbone": model.backbone},
            input_shape=(1, 3, 224, 224),
            device=device,
            num_warmup=50,
            num_runs=300,
        )

        runs[backbone] = {
            "backbone": backbone,
            "dataset": "cifar10",
            "methodology": METHODOLOGY,
            "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
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
        OUT_JSON.write_text(json.dumps(doc, indent=2))
        print(
            f"[{backbone}] acc={report.overall_accuracy:.4f} "
            f"est_flops={report.avg_computation_used:.4f} "
            f"real_speedup={cmp_real.speedup_vs('early_exit', 'backbone'):.2f}x",
            flush=True,
        )

    print("all done")


if __name__ == "__main__":
    main()
