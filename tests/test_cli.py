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
