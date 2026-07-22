"""Post-install smoke test: exercises the public API of an *installed* earlyon.

Run from OUTSIDE the repository against a clean environment:

    pip install <wheel>
    python smoke_test.py

Covers: import + version, built-in model construction, training-mode forward,
routed inference, deterministic calibration on a synthetic fixture, v2
checkpoint save/load, v1 checkpoint migration, staged runtime equivalence,
and a benchmark smoke. Uses synthetic data only — nothing here is an accuracy
or speed claim. Exits non-zero on the first failure.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        sys.exit(1)


def main() -> None:
    torch.manual_seed(0)

    import earlyon

    check("import + version", bool(earlyon.__version__), earlyon.__version__)

    # built-in model (no pretrained download), training + inference forwards
    from earlyon.models import cifar_resnet_ee

    model = cifar_resnet_ee(num_classes=10, depth=20)
    outputs = model(torch.randn(2, 3, 32, 32), mode="training")
    check("training forward", len(outputs) == len(model.config.exit_points) + 1)
    model.eval()
    result = model(torch.randn(1, 3, 32, 32), mode="inference")
    check(
        "routed inference",
        result.prediction.shape == (1, 10)
        and 0.0 <= result.estimated_backbone_flops_fraction <= 1.0,
        f"exit_taken={result.exit_taken}",
    )

    # deterministic calibration on a synthetic fixture
    from earlyon.core.thresholds import calibrate_thresholds

    x = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 10, (32,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    calib = calibrate_thresholds(model, loader, target_accuracy_drop=0.05, device="cpu")
    check(
        "calibration",
        len(calib.thresholds) == len(model.config.exit_points)
        and calib.num_samples == 32
        and calib.schema_version == 2,
        f"enabled={calib.enabled_exits}",
    )

    from earlyon.utils import load_wrapper, save_wrapper

    with tempfile.TemporaryDirectory() as tmp:
        # v2 round-trip
        v2_path = Path(tmp) / "model_v2.pth"
        save_wrapper(model, v2_path)
        loaded = load_wrapper(v2_path)
        check(
            "v2 save/load round-trip",
            loaded.config.enabled_exits == model.config.enabled_exits
            and loaded.config.temperatures == model.config.temperatures,
        )

        # v1 migration: the exact payload shape earlyon <= 0.2 wrote
        v1_path = Path(tmp) / "model_v1.pth"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": {
                    "backbone": "cifar_resnet20",
                    "num_classes": 10,
                    "routing_policy": "confidence",
                    "confidence_thresholds": [0.7, 1.0, 0.85],  # 1.0 = v1 sentinel
                    "entropy_thresholds": [0.5, 0.5, 0.5],
                    "loss_weights": [0.1, 0.2, 0.3, 0.4],
                    "temperature": 1.5,
                },
            },
            v1_path,
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            migrated = load_wrapper(v1_path)
        check(
            "v1 checkpoint migration",
            migrated.config.enabled_exits == [True, False, True]
            and migrated.config.temperatures["final"] == 1.5,
        )

    # staged runtime equivalence on a Sequential backbone
    import torch.nn as nn

    from earlyon.models import custom_ee
    from earlyon.staged import staged_model

    seq = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 10),
    )
    wrapper = custom_ee(seq, exit_layers=["0"], num_classes=10, input_shape=(1, 3, 16, 16))
    staged = staged_model(wrapper)
    wrapper.eval()
    staged.eval()
    probe = torch.randn(1, 3, 16, 16)
    eager_r = wrapper(probe, mode="inference")
    staged_r = staged.infer(probe)
    check(
        "staged == eager",
        staged_r.exit_taken == eager_r.exit_taken
        and torch.allclose(staged_r.prediction, eager_r.prediction, atol=1e-6),
    )

    # benchmark smoke: identical samples, tiny run
    from earlyon.benchmarking import benchmark_models

    cmp_r = benchmark_models(
        {"early_exit": model, "backbone": model.backbone},
        input_shape=(1, 3, 32, 32),
        num_warmup=2,
        num_runs=5,
    )
    check(
        "benchmark smoke",
        cmp_r.results["early_exit"].num_runs == 5
        and cmp_r.results["early_exit"].boundary == "model-only",
        f"speedup={cmp_r.speedup_vs('early_exit', 'backbone'):.2f}x (synthetic; not a claim)",
    )

    print("SMOKE OK")


if __name__ == "__main__":
    main()
