"""Profile a trained early-exit model on Jetson hardware.

Run on the Jetson device:

    python examples/02_jetson_deployment.py --model resnet50_ee_cifar10.pth

Outputs per-run latency, GPU utilization, power draw, and temperature.
On a non-Jetson host the script still runs but power/temp fields are zero.
"""

import argparse
import json
import statistics

from earlyon.benchmarking import JetsonProfiler, benchmark_backbone, benchmark_wrapper
from earlyon.utils import load_wrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--input-size", type=int, default=224)
    args = parser.parse_args()

    model = load_wrapper(args.model)
    shape = (1, 3, args.input_size, args.input_size)

    print("throughput: wrapper vs backbone (cuda)")
    wrap_r = benchmark_wrapper(model, input_shape=shape, device="cuda",
                                num_warmup=args.warmup, num_runs=args.runs)
    bb_r = benchmark_backbone(model.backbone, input_shape=shape, device="cuda",
                               num_warmup=args.warmup, num_runs=args.runs)
    speedup = wrap_r.throughput_ips / bb_r.throughput_ips
    print(f"  wrapper:  {wrap_r.throughput_ips:.1f} ips  "
          f"(p50 {wrap_r.latency_p50_ms:.2f}ms, p95 {wrap_r.latency_p95_ms:.2f}ms)")
    print(f"  backbone: {bb_r.throughput_ips:.1f} ips  "
          f"(p50 {bb_r.latency_p50_ms:.2f}ms, p95 {bb_r.latency_p95_ms:.2f}ms)")
    print(f"  speedup:  {speedup:.2f}x")
    print(f"  avg compute used: {wrap_r.avg_computation_used:.2%}")
    print(f"  exit distribution: {wrap_r.exit_distribution}")

    print("\njetson power/thermal profile")
    profiler = JetsonProfiler()
    runs = profiler.profile(model, input_shape=shape, num_runs=args.runs)
    print(json.dumps({
        "tegrastats_available": profiler.monitor.available,
        "latency_median_ms": statistics.median(r.latency_ms for r in runs),
        "power_median_mw": statistics.median(r.power_mw for r in runs),
        "temp_median_c": statistics.median(r.temp_c for r in runs),
    }, indent=2))


if __name__ == "__main__":
    main()
