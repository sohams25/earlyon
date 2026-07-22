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


def test_config_rejects_num_classes_below_two():
    """num_classes=1 makes softmax confidence identically 1.0, so every exit
    fires at any threshold; num_classes<1 is nonsense. Both must raise."""
    import pytest

    from earlyon.core.types import EarlyExitConfig, ExitPoint

    exits = [ExitPoint("e0", "stage1", 8)]
    for bad in (1, 0, -3):
        with pytest.raises(ValueError, match="num_classes"):
            EarlyExitConfig(backbone="tiny", num_classes=bad, exit_points=exits)


# ---------------- centralized validation (v0.3) ----------------


def test_config_migrates_legacy_scalar_temperature():
    """The legacy scalar `temperature` is broadcast deterministically onto
    every head (exits + final) at construction."""
    cfg = EarlyExitConfig(
        backbone="tiny", num_classes=10, exit_points=_two_exits(), temperature=2.0
    )
    assert cfg.temperatures == {"e0": 2.0, "e1": 2.0, "final": 2.0}


def test_config_rejects_scalar_and_per_head_temperatures_together():
    with pytest.raises(ValueError, match="temperature"):
        EarlyExitConfig(
            backbone="tiny",
            num_classes=10,
            exit_points=_two_exits(),
            temperature=2.0,
            temperatures={"e0": 1.0, "e1": 1.0, "final": 1.0},
        )


def test_config_accepts_per_head_temperatures():
    cfg = EarlyExitConfig(
        backbone="tiny",
        num_classes=10,
        exit_points=_two_exits(),
        temperatures={"e0": 1.5, "e1": 0.8, "final": 1.1},
    )
    assert cfg.temperature_for("e1") == 0.8
    assert cfg.temperature_for("final") == 1.1
    assert cfg.head_names == ["e0", "e1", "final"]


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_config_rejects_invalid_temperatures(bad):
    with pytest.raises(ValueError, match="temperatures"):
        EarlyExitConfig(
            backbone="tiny",
            num_classes=10,
            exit_points=_two_exits(),
            temperatures={"e0": bad, "e1": 1.0, "final": 1.0},
        )


def test_config_rejects_wrong_temperature_keys():
    with pytest.raises(ValueError, match="temperatures"):
        EarlyExitConfig(
            backbone="tiny",
            num_classes=10,
            exit_points=_two_exits(),
            temperatures={"e0": 1.0, "final": 1.0},  # e1 missing
        )


def test_config_rejects_duplicate_exit_names():
    exits = [ExitPoint("e0", "stage1", 16), ExitPoint("e0", "stage2", 32)]
    with pytest.raises(ValueError, match="unique"):
        EarlyExitConfig(backbone="tiny", num_classes=10, exit_points=exits)


def test_config_rejects_duplicate_layer_names():
    exits = [ExitPoint("e0", "stage1", 16), ExitPoint("e1", "stage1", 32)]
    with pytest.raises(ValueError, match="unique"):
        EarlyExitConfig(backbone="tiny", num_classes=10, exit_points=exits)


def test_config_rejects_empty_exit_or_layer_name():
    with pytest.raises(ValueError, match="non-empty"):
        EarlyExitConfig(
            backbone="tiny",
            num_classes=10,
            exit_points=[ExitPoint("", "stage1", 16)],
        )
    with pytest.raises(ValueError, match="non-empty"):
        EarlyExitConfig(
            backbone="tiny",
            num_classes=10,
            exit_points=[ExitPoint("e0", "", 16)],
        )


def test_config_rejects_reserved_final_exit_name():
    with pytest.raises(ValueError, match="reserved"):
        EarlyExitConfig(
            backbone="tiny",
            num_classes=10,
            exit_points=[ExitPoint("final", "stage1", 16)],
        )


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan"), float("inf")])
def test_config_rejects_out_of_range_confidence_thresholds(bad):
    with pytest.raises(ValueError, match="confidence_thresholds"):
        EarlyExitConfig(
            backbone="tiny",
            num_classes=10,
            exit_points=_two_exits(),
            confidence_thresholds=[bad, 0.8],
        )


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
def test_config_rejects_invalid_entropy_thresholds(bad):
    with pytest.raises(ValueError, match="entropy_thresholds"):
        EarlyExitConfig(
            backbone="tiny",
            num_classes=10,
            exit_points=_two_exits(),
            entropy_thresholds=[bad, 0.5],
        )


def test_config_rejects_negative_or_nonfinite_loss_weights():
    for bad_weights in ([-0.1, 0.6, 0.5], [float("nan"), 0.5, 0.5], [0.0, 0.0, 0.0]):
        with pytest.raises(ValueError, match="loss_weights"):
            EarlyExitConfig(
                backbone="tiny",
                num_classes=10,
                exit_points=_two_exits(),
                loss_weights=bad_weights,
            )


def test_config_rejects_wrong_enabled_exits_length():
    with pytest.raises(ValueError, match="enabled_exits"):
        EarlyExitConfig(
            backbone="tiny",
            num_classes=10,
            exit_points=_two_exits(),
            enabled_exits=[True],
        )


def test_config_defaults_enable_every_exit():
    cfg = EarlyExitConfig(backbone="tiny", num_classes=10, exit_points=_two_exits())
    assert cfg.enabled_exits == [True, True]


def test_validate_catches_post_hoc_mutation():
    """validate() is re-callable after mutation — the wrapper and checkpoint
    loader use it as their trust boundary."""
    cfg = EarlyExitConfig(backbone="tiny", num_classes=10, exit_points=_two_exits())
    cfg.temperatures["e0"] = -3.0
    with pytest.raises(ValueError, match="temperatures"):
        cfg.validate()
