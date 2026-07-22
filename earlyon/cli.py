"""Command-line interface for earlyon."""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch

from earlyon import __version__
from earlyon.benchmarking import (
    JetsonProfiler,
    benchmark_models,
    evaluate,
)
from earlyon.core.thresholds import calibrate_thresholds, calibrate_thresholds_for_budget
from earlyon.training import (
    joint_train_backbone_and_exits,
    stage1_train_backbone,
    stage2_train_exits,
)
from earlyon.utils import build_model, cifar10_loaders, load_wrapper, save_wrapper

BACKBONES = ["resnet18", "resnet50", "mobilenetv2", "efficientnet_b0", "vit_b_16"]
_CIFAR_PREFIX = "cifar_resnet"


def _backbone_arg(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """Accept the factory names plus 'cifar_resnet<depth>' (bare 'cifar_resnet'
    maps to the factory default, depth 56). The depth rides in the name so the
    saved config round-trips through build_model."""
    if value in BACKBONES:
        return value
    if value == _CIFAR_PREFIX:
        return f"{_CIFAR_PREFIX}56"
    if value.startswith(_CIFAR_PREFIX) and value[len(_CIFAR_PREFIX) :].isdigit():
        return value
    raise click.BadParameter(
        f"choose from {BACKBONES} or 'cifar_resnet<depth>' (e.g. cifar_resnet20)"
    )


def _target_compute_arg(
    ctx: click.Context, param: click.Parameter, value: float | None
) -> float | None:
    if value is not None and not 0.0 < value <= 1.0:
        raise click.BadParameter(f"must be in (0, 1], got {value}")
    return value


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
@click.option(
    "--backbone",
    required=True,
    callback=_backbone_arg,
    metavar="[resnet18|resnet50|mobilenetv2|efficientnet_b0|vit_b_16|cifar_resnet<depth>]",
)
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
@click.option(
    "--backbone",
    required=True,
    callback=_backbone_arg,
    metavar="[resnet18|resnet50|mobilenetv2|efficientnet_b0|vit_b_16|cifar_resnet<depth>]",
)
@click.option("--num-classes", required=True, type=int)
@click.option("--dataset", default="cifar10", type=click.Choice(["cifar10"]))
@click.option("--epochs", default=90, type=int)
@click.option("--lr", default=0.1, type=float)
@click.option("--batch-size", default=128, type=int)
@click.option("--device", default="auto")
@click.option("--validate/--no-validate", default=False, help="report val metrics each epoch")
@click.option("--output", required=True, type=click.Path())
def train_backbone(
    backbone: str,
    num_classes: int,
    dataset: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    validate: bool,
    output: str,
) -> None:
    """Stage 1: train the backbone as a standard classifier."""
    dev = _device(device)
    train_loader, val_loader, _ = cifar10_loaders(batch_size=batch_size, val_batch_size=batch_size)
    model = build_model(backbone, num_classes=num_classes, pretrained=True)
    stage1_train_backbone(
        model.backbone,
        train_loader,
        val_loader=val_loader if validate else None,
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
@click.option("--validate/--no-validate", default=False, help="report val metrics each epoch")
@click.option("--output", required=True, type=click.Path())
def train_exits(
    model_path: str,
    dataset: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    validate: bool,
    output: str,
) -> None:
    """Stage 2: freeze backbone, train exit heads."""
    dev = _device(device)
    train_loader, val_loader, _ = cifar10_loaders(batch_size=batch_size, val_batch_size=batch_size)
    model = load_wrapper(model_path)
    stage2_train_exits(
        model,
        train_loader,
        val_loader=val_loader if validate else None,
        epochs=epochs,
        lr=lr,
        device=dev,
    )
    save_wrapper(model, output)
    click.echo(f"wrote {output}")


@train.command("joint")
@click.option("--model", "model_path", required=True, type=click.Path(exists=True))
@click.option("--dataset", default="cifar10", type=click.Choice(["cifar10"]))
@click.option("--epochs", default=30, type=int)
@click.option("--lr", default=1e-2, type=float)
@click.option("--batch-size", default=128, type=int)
@click.option("--device", default="auto")
@click.option("--validate/--no-validate", default=False, help="report val metrics each epoch")
@click.option("--output", required=True, type=click.Path())
def train_joint(
    model_path: str,
    dataset: str,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    validate: bool,
    output: str,
) -> None:
    """Joint training: backbone and exit heads update together."""
    dev = _device(device)
    train_loader, val_loader, _ = cifar10_loaders(batch_size=batch_size, val_batch_size=batch_size)
    model = load_wrapper(model_path)
    joint_train_backbone_and_exits(
        model,
        train_loader,
        val_loader=val_loader if validate else None,
        epochs=epochs,
        lr=lr,
        device=dev,
    )
    save_wrapper(model, output)
    click.echo(f"wrote {output}")


@main.command()
@click.option("--model", "model_path", required=True, type=click.Path(exists=True))
@click.option("--dataset", default="cifar10", type=click.Choice(["cifar10"]))
@click.option("--target-drop", default=0.01, type=float)
@click.option(
    "--target-compute",
    default=None,
    type=float,
    callback=_target_compute_arg,
    help="calibrate to a compute budget instead of an accuracy budget: the "
    "average FLOPs fraction the deployed model may use, in (0, 1]. "
    "Overrides --target-drop.",
)
@click.option("--device", default="auto")
@click.option("--output", required=True, type=click.Path())
def calibrate(
    model_path: str,
    dataset: str,
    target_drop: float,
    target_compute: float | None,
    device: str,
    output: str,
) -> None:
    """Greedy threshold calibration on the validation split."""
    dev = _device(device)
    _, val_loader, _ = cifar10_loaders(batch_size=1)
    model = load_wrapper(model_path)
    if target_compute is not None:
        result = calibrate_thresholds_for_budget(
            model, val_loader, target_computation=target_compute, device=dev
        )
    else:
        result = calibrate_thresholds(
            model, val_loader, target_accuracy_drop=target_drop, device=dev
        )
    save_wrapper(model, output)
    line = (
        f"thresholds={result.thresholds} baseline_acc={result.baseline_accuracy:.4f} "
        f"final_acc={result.final_accuracy:.4f} avg_comp={result.avg_computation_used:.4f}"
    )
    if target_compute is not None:
        line += f" budget_met={result.budget_met}"
    click.echo(line)


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
    """Throughput + latency benchmark, single-sample (batch=1).

    The wrapper and its raw backbone are measured on the exact same
    fixed-seed noise samples with identical boundaries. Noise input is a
    best-case bound (trained heads may fire spuriously); use the Python API
    with a real loader for the honest input-distribution signal.
    """
    dev = _device(device)
    model = load_wrapper(model_path)
    shape = (1, 3, input_size, input_size)
    cmp_r = benchmark_models(
        {"early_exit": model, "backbone": model.backbone},
        input_shape=shape,
        device=dev,
        num_warmup=warmup,
        num_runs=runs,
    )
    wrap_r = cmp_r.results["early_exit"]
    bb_r = cmp_r.results["backbone"]
    speedup = cmp_r.speedup_vs("early_exit", "backbone")
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
    """Jetson profile — reads tegrastats when available.

    Reports instantaneous-power medians and, separately, integrated energy
    over the timed window. Missing telemetry (non-Jetson host) is reported as
    null, never as zero.
    """
    dev = _device(device)
    model = load_wrapper(model_path)
    profiler = JetsonProfiler()
    runs_out, energy = profiler.profile_with_energy(
        model,
        input_shape=(1, 3, input_size, input_size),
        num_warmup=warmup,
        num_runs=runs,
        device=dev,
    )
    import statistics

    lat = [r.latency_ms for r in runs_out]
    power = [r.power_mw for r in runs_out if r.power_mw is not None]
    temp = [r.temp_c for r in runs_out if r.temp_c is not None]
    click.echo(
        json.dumps(
            {
                "runs": len(runs_out),
                "latency_median_ms": statistics.median(lat),
                "instantaneous_power_median_mw": statistics.median(power) if power else None,
                "temp_median_c": statistics.median(temp) if temp else None,
                "energy": {
                    "window_seconds": energy.window_seconds,
                    "num_power_samples": energy.num_power_samples,
                    "avg_power_mw": energy.avg_power_mw,
                    "energy_mj": energy.energy_mj,
                    "energy_per_inference_mj": energy.energy_per_inference_mj,
                },
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


@main.command()
@click.option("--model", "model_path", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
@click.option("--input-size", default=224, type=int)
@click.option("--opset", default=17, type=int)
@click.option("--dynamic-batch/--no-dynamic-batch", default=True)
def export(model_path: str, output: str, input_size: int, opset: int, dynamic_batch: bool) -> None:
    """Export all exits as a static multi-output ONNX graph (routing at runtime)."""
    from earlyon.onnx import export_to_onnx

    model = load_wrapper(model_path)
    names = export_to_onnx(
        model,
        output,
        input_shape=(1, 3, input_size, input_size),
        opset=opset,
        dynamic_batch=dynamic_batch,
    )
    click.echo(f"wrote {output} (input 1x3x{input_size}x{input_size}, outputs: {', '.join(names)})")


if __name__ == "__main__":
    main()
