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


def test_train_subcommands_exist():
    runner = CliRunner()
    result = runner.invoke(main, ["train", "--help"])
    assert result.exit_code == 0
    assert "backbone" in result.output
    assert "exits" in result.output


def test_wrap_creates_checkpoint(tmp_path):
    runner = CliRunner()
    out = tmp_path / "m.pth"
    result = runner.invoke(
        main,
        ["wrap", "--backbone", "resnet18", "--num-classes", "10",
         "--no-pretrained", "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_save_load_round_trip(tmp_path):
    """save_wrapper + load_wrapper must preserve thresholds, weights, temperature."""
    import torch

    from earlyon.utils import build_model, load_wrapper, save_wrapper

    model = build_model("resnet18", num_classes=10, pretrained=False)
    model.config.confidence_thresholds = [0.42, 0.73]
    model.config.loss_weights = [0.1, 0.2, 0.7]
    model.config.temperature = 1.5
    # mutate an exit head weight so we can verify state_dict round-trip
    with torch.no_grad():
        first_param = next(iter(model.exit_heads["e0"].parameters()))
        first_param.fill_(0.123)

    path = tmp_path / "rt.pth"
    save_wrapper(model, path)
    loaded = load_wrapper(path)

    assert loaded.config.confidence_thresholds == [0.42, 0.73]
    assert loaded.config.loss_weights == [0.1, 0.2, 0.7]
    assert loaded.config.temperature == 1.5
    loaded_first = next(iter(loaded.exit_heads["e0"].parameters()))
    assert torch.allclose(loaded_first, torch.full_like(loaded_first, 0.123))
