"""Smoke tests for the CLI. We don't run training (too slow) — just import
the click app and check the command tree is registered."""

from click.testing import CliRunner

from earlyon.cli import main


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0


def test_cli_help_lists_commands():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in ["wrap", "train", "calibrate", "benchmark", "profile", "analyze"]:
        assert cmd in result.output


def test_calibrate_exposes_target_compute_option():
    runner = CliRunner()
    result = runner.invoke(main, ["calibrate", "--help"])
    assert result.exit_code == 0
    assert "--target-compute" in result.output
    assert "--target-drop" in result.output


def test_train_subcommands_exist():
    runner = CliRunner()
    result = runner.invoke(main, ["train", "--help"])
    assert result.exit_code == 0
    assert "backbone" in result.output
    assert "exits" in result.output
    assert "joint" in result.output


def test_train_exits_validate_flag_threads_val_loader(tmp_path, monkeypatch):
    """`--validate` must pass a real val_loader to the trainer (None otherwise),
    and the CLI must request a batched val loader (val_batch_size == batch_size)."""
    import earlyon.cli as cli

    captured: dict = {}
    sentinel_train, sentinel_val = object(), object()

    def fake_loaders(batch_size=128, val_batch_size=1, **kw):
        captured["val_batch_size"] = val_batch_size
        return sentinel_train, sentinel_val, object()

    def fake_stage2(model, train_loader, val_loader=None, **kw):
        captured["val_loader"] = val_loader
        return model

    monkeypatch.setattr(cli, "cifar10_loaders", fake_loaders)
    monkeypatch.setattr(cli, "load_wrapper", lambda path, **kw: "MODEL")
    monkeypatch.setattr(cli, "stage2_train_exits", fake_stage2)
    monkeypatch.setattr(cli, "save_wrapper", lambda model, path: None)

    model_path = tmp_path / "in.pth"
    model_path.write_text("x")
    out = tmp_path / "out.pth"
    runner = CliRunner()

    r1 = runner.invoke(
        cli.main,
        [
            "train",
            "exits",
            "--model",
            str(model_path),
            "--batch-size",
            "32",
            "--epochs",
            "1",
            "--validate",
            "--output",
            str(out),
        ],
    )
    assert r1.exit_code == 0, r1.output
    assert captured["val_loader"] is sentinel_val
    assert captured["val_batch_size"] == 32

    r2 = runner.invoke(
        cli.main,
        [
            "train",
            "exits",
            "--model",
            str(model_path),
            "--epochs",
            "1",
            "--no-validate",
            "--output",
            str(out),
        ],
    )
    assert r2.exit_code == 0, r2.output
    assert captured["val_loader"] is None


def test_wrap_creates_checkpoint(tmp_path):
    runner = CliRunner()
    out = tmp_path / "m.pth"
    result = runner.invoke(
        main,
        [
            "wrap",
            "--backbone",
            "resnet18",
            "--num-classes",
            "10",
            "--no-pretrained",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_save_load_round_trip(tmp_path):
    """save_wrapper + load_wrapper must preserve thresholds, weights,
    per-head temperatures, and enabled_exits."""
    import torch

    from earlyon.utils import build_model, load_wrapper, save_wrapper

    model = build_model("resnet18", num_classes=10, pretrained=False)
    model.config.confidence_thresholds = [0.42, 0.73]
    model.config.loss_weights = [0.1, 0.2, 0.7]
    model.config.temperatures = {"e0": 1.5, "e1": 0.9, "final": 1.2}
    model.config.enabled_exits = [True, False]
    # mutate an exit head weight so we can verify state_dict round-trip
    with torch.no_grad():
        first_param = next(iter(model.exit_heads["e0"].parameters()))
        first_param.fill_(0.123)

    path = tmp_path / "rt.pth"
    save_wrapper(model, path)
    loaded = load_wrapper(path)

    assert loaded.config.confidence_thresholds == [0.42, 0.73]
    assert loaded.config.loss_weights == [0.1, 0.2, 0.7]
    assert loaded.config.temperatures == {"e0": 1.5, "e1": 0.9, "final": 1.2}
    assert loaded.config.enabled_exits == [True, False]
    loaded_first = next(iter(loaded.exit_heads["e0"].parameters()))
    assert torch.allclose(loaded_first, torch.full_like(loaded_first, 0.123))


def test_save_load_round_trip_preserves_entropy_routing(tmp_path):
    """Regression: an entropy-routed model must reload as entropy-routed with its
    calibrated entropy_thresholds intact. Before persisting these fields, reload
    silently reverted to confidence routing and dropped the entropy thresholds."""
    from earlyon.utils import build_model, load_wrapper, save_wrapper

    model = build_model("resnet18", num_classes=10, pretrained=False)
    model.config.routing_policy = "entropy"
    model.config.entropy_thresholds = [0.3, 0.6]
    model.config.confidence_thresholds = [0.85, 0.80]

    path = tmp_path / "ent.pth"
    save_wrapper(model, path)
    loaded = load_wrapper(path)

    assert loaded.config.routing_policy == "entropy"
    assert loaded.config.entropy_thresholds == [0.3, 0.6]
    assert loaded.config.confidence_thresholds == [0.85, 0.80]


def test_load_wrapper_back_compat_without_entropy_fields(tmp_path):
    """Pre-0.2 checkpoints have no routing_policy/entropy_thresholds keys; load
    must not raise and must fall back to the fresh model's defaults."""
    import torch

    from earlyon.utils import build_model, load_wrapper

    model = build_model("resnet18", num_classes=10, pretrained=False)
    path = tmp_path / "old.pth"
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "backbone": "resnet18",
            "num_classes": 10,
            "confidence_thresholds": [0.7, 0.8],
            "loss_weights": [0.2, 0.3, 0.5],
            "temperature": 1.0,
            # no routing_policy / entropy_thresholds (pre-0.2)
        },
    }
    torch.save(payload, path)

    loaded = load_wrapper(path)  # must not raise
    assert loaded.config.routing_policy == "confidence"
    assert len(loaded.config.entropy_thresholds) == 2


def test_save_load_round_trip_cifar_resnet(tmp_path):
    """Regression: cifar_resnet_ee records its depth in the backbone string
    ('cifar_resnet20'); build_model must reconstruct it so the checkpoint loads.
    Previously any cifar_resnet checkpoint raised 'unknown backbone'."""
    from earlyon.models import cifar_resnet_ee
    from earlyon.utils import load_wrapper, save_wrapper

    model = cifar_resnet_ee(num_classes=10, depth=20)
    n_exits = len(model.config.exit_points)
    model.config.confidence_thresholds = [0.7] * n_exits

    path = tmp_path / "cifar.pth"
    save_wrapper(model, path)
    loaded = load_wrapper(path)  # must not raise

    assert loaded.config.backbone == "cifar_resnet20"
    assert loaded.config.confidence_thresholds == [0.7] * n_exits
    assert len(loaded.config.exit_points) == n_exits


def test_build_model_rejects_malformed_cifar_backbone():
    import pytest

    from earlyon.utils import build_model

    with pytest.raises(ValueError, match="malformed cifar_resnet"):
        build_model("cifar_resnetXYZ", num_classes=10)


def test_load_wrapper_rejects_entropy_policy_without_thresholds(tmp_path):
    """A checkpoint claiming entropy routing but missing entropy_thresholds must
    raise rather than silently route on uncalibrated defaults."""
    import pytest
    import torch

    from earlyon.utils import build_model, load_wrapper

    model = build_model("resnet18", num_classes=10, pretrained=False)
    path = tmp_path / "ent_missing.pth"
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "backbone": "resnet18",
            "num_classes": 10,
            "routing_policy": "entropy",  # claims entropy...
            "confidence_thresholds": [0.85, 0.80],
            # ...but no entropy_thresholds key
            "loss_weights": [0.2, 0.3, 0.5],
            "temperature": 1.0,
        },
    }
    torch.save(payload, path)
    with pytest.raises(ValueError, match="entropy_thresholds"):
        load_wrapper(path)


def test_load_wrapper_rejects_invalid_routing_policy(tmp_path):
    """A corrupted checkpoint with an unknown routing_policy must raise on load
    rather than silently mis-route (load bypasses EarlyExitConfig validation)."""
    import pytest
    import torch

    from earlyon.utils import build_model, load_wrapper

    model = build_model("resnet18", num_classes=10, pretrained=False)
    path = tmp_path / "badpolicy.pth"
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "backbone": "resnet18",
            "num_classes": 10,
            "routing_policy": "budget",  # not a supported policy
            "confidence_thresholds": [0.85, 0.80],
            "entropy_thresholds": [0.5, 0.5],
            "loss_weights": [0.2, 0.3, 0.5],
            "temperature": 1.0,
        },
    }
    torch.save(payload, path)

    with pytest.raises(ValueError, match="routing_policy"):
        load_wrapper(path)


def test_load_wrapper_rejects_mismatched_threshold_count(tmp_path):
    """Saved checkpoint with N+1 thresholds vs N-exit factory must raise
    ValueError before mutating model.config."""
    import torch

    from earlyon.utils import build_model, load_wrapper

    model = build_model("resnet18", num_classes=10, pretrained=False)
    # resnet18 has 2 exit points; save a corrupted config with 3 thresholds
    path = tmp_path / "bad.pth"
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "backbone": "resnet18",
            "num_classes": 10,
            "confidence_thresholds": [0.7, 0.8, 0.9],  # wrong length: should be 2
            "loss_weights": [0.2, 0.3, 0.5],
            "temperature": 1.0,
        },
    }
    torch.save(payload, path)

    import pytest

    with pytest.raises(ValueError, match="confidence_thresholds has length 3"):
        load_wrapper(path)


def test_load_wrapper_rejects_mismatched_loss_weights(tmp_path):
    import torch

    from earlyon.utils import build_model, load_wrapper

    model = build_model("resnet18", num_classes=10, pretrained=False)
    path = tmp_path / "bad.pth"
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "backbone": "resnet18",
            "num_classes": 10,
            "confidence_thresholds": [0.7, 0.8],
            "loss_weights": [0.5, 0.5],  # wrong length: should be 3
            "temperature": 1.0,
        },
    }
    torch.save(payload, path)

    import pytest

    with pytest.raises(ValueError, match="loss_weights has length 2"):
        load_wrapper(path)


def test_wrap_accepts_cifar_resnet_backbone(tmp_path):
    """The CIFAR-native backbone from the README models table must be
    wrappable from the CLI, with the depth encoded in the name."""
    runner = CliRunner()
    out = tmp_path / "m.pth"
    result = runner.invoke(
        main,
        ["wrap", "--backbone", "cifar_resnet20", "--num-classes", "10", "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_calibrate_rejects_out_of_range_target_compute_cleanly(tmp_path):
    """--target-compute outside (0, 1] must fail at option parsing with a
    click error (exit code 2, no traceback), before any dataset is touched."""
    dummy = tmp_path / "dummy.pth"
    dummy.write_bytes(b"not a real checkpoint")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["calibrate", "--model", str(dummy), "--target-compute", "1.5", "--output", "o.pth"],
    )
    assert result.exit_code == 2
    assert "target-compute" in result.output or "target_compute" in result.output
    assert "Traceback" not in result.output
