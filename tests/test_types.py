import pytest

from earlyon.core.types import EarlyExitConfig, ExitPoint


def _two_exits():
    return [ExitPoint("e0", "stage1", 16), ExitPoint("e1", "stage2", 32)]


def test_default_loss_weights_sum_to_one():
    cfg = EarlyExitConfig(backbone="tiny", num_classes=10, exit_points=_two_exits())
    assert len(cfg.loss_weights) == 3  # 2 exits + final
    assert abs(sum(cfg.loss_weights) - 1.0) < 1e-9


def test_default_thresholds_match_exit_count():
    cfg = EarlyExitConfig(backbone="tiny", num_classes=10, exit_points=_two_exits())
    assert len(cfg.confidence_thresholds) == 2


def test_wrong_loss_weight_length_raises():
    with pytest.raises(ValueError, match="loss_weights"):
        EarlyExitConfig(
            backbone="tiny",
            num_classes=10,
            exit_points=_two_exits(),
            loss_weights=[0.5, 0.5],  # should be length 3
        )


def test_wrong_threshold_length_raises():
    with pytest.raises(ValueError, match="confidence_thresholds"):
        EarlyExitConfig(
            backbone="tiny",
            num_classes=10,
            exit_points=_two_exits(),
            confidence_thresholds=[0.8],
        )


def test_unsupported_routing_policy_raises():
    with pytest.raises(ValueError, match="routing_policy"):
        EarlyExitConfig(
            backbone="tiny",
            num_classes=10,
            exit_points=_two_exits(),
            routing_policy="budget",  # not implemented yet
        )


def test_entropy_routing_policy_accepted():
    cfg = EarlyExitConfig(
        backbone="tiny",
        num_classes=10,
        exit_points=_two_exits(),
        routing_policy="entropy",
    )
    # entropy_thresholds default to a per-exit value matching exit count
    assert len(cfg.entropy_thresholds) == 2
    assert all(t > 0 for t in cfg.entropy_thresholds)


def test_entropy_threshold_length_validated():
    with pytest.raises(ValueError, match="entropy_thresholds"):
        EarlyExitConfig(
            backbone="tiny",
            num_classes=10,
            exit_points=_two_exits(),
            routing_policy="entropy",
            entropy_thresholds=[0.5],  # should be length 2
        )
