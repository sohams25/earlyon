"""Command-line interface for earlyon."""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch

from earlyon import __version__
from earlyon.benchmarking import (
    JetsonProfiler,
    benchmark_backbone,
    benchmark_wrapper,
    evaluate,
)
from earlyon.core.thresholds import calibrate_thresholds
from earlyon.training import stage1_train_backbone, stage2_train_exits
from earlyon.utils import build_model, cifar10_loaders, load_wrapper, save_wrapper

BACKBONES = ["resnet18", "resnet50", "mobilenetv2", "efficientnet_b0"]


def _device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "jetson":
        return "cuda"
    return name


@click.group()
@click.version_option(__version__)
def main() -> None:
    """earlyon — early exit toolkit for PyTorch CV models."""


@main.command()
@click.option("--backbone", required=True, type=click.Choice(BACKBONES))
@click.option("--num-classes", required=True, type=int)
@click.option("--pretrained/--no-pretrained", default=True)
@click.option("--output", "output", required=True, type=click.Path())
def wrap(backbone: str, num_classes: int, pretrained: bool, output: str) -> None:
    """Build a fresh early-exit wrapper and save its state_dict + config."""
    model = build_model(backbone, num_classes=num_classes, pretrained=pretrained)
    save_wrapper(model, output)
    click.echo(f"wrote {output} ({backbone}, num_classes={num_classes})")


@main.group()
def train() -> None:
    """Two-stage training commands."""


@train.command("backbone")
@click.option("--backbone", required=True, type=click.Choice(BACKBONES))
@click.option("--num-classes", required=True, type=int)
@click.option("--dataset", default="cifar10", type=click.Choice(["cifar10"]))
@click.option("--epochs", default=90, type=int)
@click.option("--lr", default=0.1, type=float)
@click.option("--batch-size", default=128, type=int)
@click.option("--device", default="auto")
@click.option("--output", required=True, type=click.Path())
def train_backbone(
    backbone: str,
    num_classes: int,
    dataset: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    output: str,
) -> None:
    """Stage 1: train the backbone as a standard classifier."""
    dev = _device(device)
    train_loader, _, _ = cifar10_loaders(batch_size=batch_size)
    model = build_model(backbone, num_classes=num_classes, pretrained=True)
    stage1_train_backbone(
        model.backbone,
        train_loader,
        epochs=epochs,
        lr=lr,
        device=dev,
    )
    save_wrapper(model, output)
    click.echo(f"wrote {output}")


@train.command("exits")
@click.option("--model", "model_path", required=True, type=click.Path(exists=True))
@click.option("--dataset", default="cifar10", type=click.Choice(["cifar10"]))
@click.option("--epochs", default=20, type=int)
@click.option("--lr", default=1e-3, type=float)
@click.option("--batch-size", default=128, type=int)
@click.option("--device", default="auto")
@click.option("--output", required=True, type=click.Path())
def train_exits(
    model_path: str,
    dataset: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    output: str,
) -> None:
    """Stage 2: freeze backbone, train exit heads."""
    dev = _device(device)
    train_loader, _, _ = cifar10_loaders(batch_size=batch_size)
    model = load_wrapper(model_path)
    stage2_train_exits(model, train_loader, epochs=epochs, lr=lr, device=dev)
    save_wrapper(model, output)
    click.echo(f"wrote {output}")


@main.command()
@click.option("--model", "model_path", required=True, type=click.Path(exists=True))
@click.option("--dataset", default="cifar10", type=click.Choice(["cifar10"]))
@click.option("--target-drop", default=0.01, type=float)
@click.option("--device", default="auto")
@click.option("--output", required=True, type=click.Path())
def calibrate(model_path: str, dataset: str, target_drop: float, device: str, output: str) -> None:
    """Greedy threshold calibration on the validation split."""
    dev = _device(device)
    _, val_loader, _ = cifar10_loaders(batch_size=1)
    model = load_wrapper(model_path)
    result = calibrate_thresholds(model, val_loader, target_accuracy_drop=target_drop, device=dev)
    save_wrapper(model, output)
    click.echo(
        f"thresholds={result.thresholds} baseline_acc={result.baseline_accuracy:.4f} "
        f"final_acc={result.final_accuracy:.4f} avg_comp={result.avg_computation_used:.4f}"
    )


@main.command()
@click.option("--model", "model_path", required=True, type=click.Path(exists=True))
@click.option("--device", default="auto")
@click.option("--runs", default=500, type=int)
@click.option("--warmup", default=50, type=int)
@click.option("--input-size", default=224, type=int)
@click.option("--json-out", default=None, type=click.Path())
def benchmark(
    model_path: str, device: str, runs: int, warmup: int, input_size: int, json_out: str | None
) -> None:
    """Throughput + latency benchmark, single-sample (batch=1)."""
    dev = _device(device)
    model = load_wrapper(model_path)
    shape = (1, 3, input_size, input_size)
    wrap_r = benchmark_wrapper(
        model, input_shape=shape, device=dev, num_warmup=warmup, num_runs=runs
    )
    bb_r = benchmark_backbone(
        model.backbone, input_shape=shape, device=dev, num_warmup=warmup, num_runs=runs
    )
    speedup = wrap_r.throughput_ips / max(bb_r.throughput_ips, 1e-9)
    click.echo(
        json.dumps(
            {
                "device": dev,
                "speedup": speedup,
                "wrapper": wrap_r.to_dict(),
                "backbone": bb_r.to_dict(),
            },
            indent=2,
        )
    )
    if json_out:
        Path(json_out).write_text(
            json.dumps(
                {
                    "speedup": speedup,
                    "wrapper": wrap_r.to_dict(),
                    "backbone": bb_r.to_dict(),
                },
                indent=2,
            )
        )


@main.command()
@click.option("--model", "model_path", required=True, type=click.Path(exists=True))
@click.option("--runs", default=100, type=int)
@click.option("--warmup", default=50, type=int)
@click.option("--input-size", default=224, type=int)
@click.option("--device", default="auto")
def profile(model_path: str, runs: int, warmup: int, input_size: int, device: str) -> None:
    """Jetson profile — reads tegrastats when available."""
    dev = _device(device)
    model = load_wrapper(model_path)
    profiler = JetsonProfiler()
    runs_out = profiler.profile(
        model,
        input_shape=(1, 3, input_size, input_size),
        num_warmup=warmup,
        num_runs=runs,
        device=dev,
    )
    import statistics

    lat = [r.latency_ms for r in runs_out]
    power = [r.power_mw for r in runs_out]
    temp = [r.temp_c for r in runs_out]
    click.echo(
        json.dumps(
            {
                "runs": len(runs_out),
                "latency_median_ms": statistics.median(lat),
                "power_median_mw": statistics.median(power),
                "temp_median_c": statistics.median(temp),
                "tegrastats_available": profiler.monitor.available,
            },
            indent=2,
        )
    )


@main.command()
@click.option("--model", "model_path", required=True, type=click.Path(exists=True))
@click.option("--dataset", default="cifar10", type=click.Choice(["cifar10"]))
@click.option("--device", default="auto")
def analyze(model_path: str, dataset: str, device: str) -> None:
    """Per-exit accuracy + exit distribution on the test set."""
    dev = _device(device)
    _, _, test_loader = cifar10_loaders(batch_size=1)
    model = load_wrapper(model_path)
    report = evaluate(model, test_loader, device=dev)
    click.echo(
        json.dumps(
            {
                "overall_accuracy": report.overall_accuracy,
                "avg_computation_used": report.avg_computation_used,
                "exit_distribution": report.exit_distribution,
                "per_exit_accuracy": report.per_exit_accuracy,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
