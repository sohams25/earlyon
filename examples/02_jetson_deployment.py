"""Profile a trained early-exit model on Jetson hardware.

Run on the Jetson device:

    python examples/02_jetson_deployment.py --model resnet50_ee_cifar10.pth

Outputs a fair wrapper-vs-backbone throughput comparison (identical noise
samples and boundaries for both models — a best-case bound, since trained
heads may fire spuriously on noise), then per-run latency with instantaneous
power/temperature and the energy integrated over the timed window.
On a non-Jetson host the script still runs; missing telemetry is reported as
null, never as zero.
"""

import argparse
import json
import statistics

from earlyon.benchmarking import JetsonProfiler, benchmark_models
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

    print("fair throughput: wrapper vs backbone on identical samples (cuda)")
    cmp_r = benchmark_models(
        {"early_exit": model, "backbone": model.backbone},
        input_shape=shape, device="cuda",
        num_warmup=args.warmup, num_runs=args.runs,
    )
    wrap_r = cmp_r.results["early_exit"]
    bb_r = cmp_r.results["backbone"]
    print(f"  wrapper:  {wrap_r.throughput_ips:.1f} ips  "
          f"(p50 {wrap_r.latency_p50_ms:.2f}ms, p95 {wrap_r.latency_p95_ms:.2f}ms)")
    print(f"  backbone: {bb_r.throughput_ips:.1f} ips  "
          f"(p50 {bb_r.latency_p50_ms:.2f}ms, p95 {bb_r.latency_p95_ms:.2f}ms)")
    print(f"  speedup:  {cmp_r.speedup_vs('early_exit', 'backbone'):.2f}x "
          f"(noise input: best-case bound)")
    print(f"  est. backbone FLOPs fraction: {wrap_r.avg_estimated_flops_fraction:.2%}")
    print(f"  exit distribution: {wrap_r.exit_distribution}")

    print("\njetson power/thermal/energy profile")
    profiler = JetsonProfiler()
    runs, energy = profiler.profile_with_energy(
        model, input_shape=shape, num_warmup=args.warmup, num_runs=args.runs
    )
    power = [r.power_mw for r in runs if r.power_mw is not None]
    temp = [r.temp_c for r in runs if r.temp_c is not None]
    print(json.dumps({
        "tegrastats_available": profiler.monitor.available,
        "latency_median_ms": statistics.median(r.latency_ms for r in runs),
        "instantaneous_power_median_mw": statistics.median(power) if power else None,
        "temp_median_c": statistics.median(temp) if temp else None,
        "avg_power_mw": energy.avg_power_mw,
        "energy_mj": energy.energy_mj,
        "energy_per_inference_mj": energy.energy_per_inference_mj,
        "num_power_samples": energy.num_power_samples,
    }, indent=2))


if __name__ == "__main__":
    main()
