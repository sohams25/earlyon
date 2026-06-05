import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from earlyon.core.exit_head import EarlyExitHead
from earlyon.core.types import EarlyExitConfig, ExitPoint
from earlyon.core.wrappers import EarlyExitWrapper
from earlyon.training import (
    joint_train_backbone_and_exits,
    stage1_train_backbone,
    stage2_train_exits,
    weighted_multi_exit_loss,
)
from tests.fixtures.tiny_models import STAGE_CHANNELS, TinyBackbone


def _toy_loader(n=32, n_classes=10, shape=(3, 32, 32), bs=8):
    x = torch.randn(n, *shape)
    y = torch.randint(0, n_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=bs)


def _build_wrapper():
    backbone = TinyBackbone(num_classes=10)
    exits = [
        ExitPoint("e0", "stage1", STAGE_CHANNELS["stage1"]),
        ExitPoint("e1", "stage2", STAGE_CHANNELS["stage2"]),
    ]
    heads = {ep.name: EarlyExitHead(ep.in_channels, 10) for ep in exits}
    cfg = EarlyExitConfig(
        backbone="tiny",
        num_classes=10,
        exit_points=exits,
        confidence_thresholds=[0.8, 0.8],
    )
    return EarlyExitWrapper(backbone, heads, lambda x: x, cfg, input_shape=(1, 3, 32, 32))


def test_weighted_loss_value_matches_manual_sum():
    pred1 = torch.randn(4, 10, requires_grad=True)
    pred2 = torch.randn(4, 10, requires_grad=True)
    targets = torch.randint(0, 10, (4,))
    weights = [0.3, 0.7]
    loss = weighted_multi_exit_loss([pred1, pred2], targets, weights)
    manual = 0.3 * F.cross_entropy(pred1, targets) + 0.7 * F.cross_entropy(pred2, targets)
    assert torch.allclose(loss, manual)


def test_weighted_loss_length_mismatch_raises():
    with pytest.raises(ValueError, match="predictions but"):
        weighted_multi_exit_loss([torch.randn(2, 5)], torch.zeros(2, dtype=torch.long), [0.5, 0.5])


def test_stage1_runs_one_epoch():
    bb = TinyBackbone(num_classes=10)
    loader = _toy_loader()
    out = stage1_train_backbone(
        bb, loader, epochs=1, lr=0.01, device="cpu", on_epoch_end=lambda _: None
    )
    assert out is bb


def test_stage1_reports_validation_metrics():
    """A val_loader makes stage 1 report per-epoch val loss/accuracy."""
    bb = TinyBackbone(num_classes=10)
    logs: list = []
    stage1_train_backbone(
        bb,
        _toy_loader(n=16),
        val_loader=_toy_loader(n=8),
        epochs=1,
        lr=0.01,
        device="cpu",
        on_epoch_end=logs.append,
    )
    assert len(logs) == 1
    assert logs[0].val_loss is not None and logs[0].val_loss >= 0.0
    assert logs[0].val_accuracy is not None and 0.0 <= logs[0].val_accuracy <= 1.0


def test_stage2_reports_validation_metrics():
    wrapper = _build_wrapper()
    logs: list = []
    stage2_train_exits(
        wrapper,
        _toy_loader(n=16),
        val_loader=_toy_loader(n=8),
        epochs=1,
        device="cpu",
        on_epoch_end=logs.append,
    )
    assert logs[0].val_loss is not None and logs[0].val_loss >= 0.0
    assert logs[0].val_accuracy is not None and 0.0 <= logs[0].val_accuracy <= 1.0


def test_joint_reports_validation_metrics():
    wrapper = _build_wrapper()
    logs: list = []
    joint_train_backbone_and_exits(
        wrapper,
        _toy_loader(n=16),
        val_loader=_toy_loader(n=8),
        epochs=1,
        device="cpu",
        on_epoch_end=logs.append,
    )
    assert logs[0].val_loss is not None and logs[0].val_loss >= 0.0
    assert logs[0].val_accuracy is not None and 0.0 <= logs[0].val_accuracy <= 1.0


@pytest.mark.parametrize("which", ["stage1", "stage2", "joint"])
def test_no_val_loader_leaves_metrics_none_without_warning(which):
    """For every trainer, omitting val_loader leaves val metrics None and emits
    no warning (the removed _warn_if_val_loader_unused is fully replaced)."""
    import warnings

    logs: list = []
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if which == "stage1":
            stage1_train_backbone(
                TinyBackbone(num_classes=10),
                _toy_loader(n=16),
                epochs=1,
                lr=0.01,
                device="cpu",
                on_epoch_end=logs.append,
            )
        elif which == "stage2":
            stage2_train_exits(
                _build_wrapper(),
                _toy_loader(n=16),
                epochs=1,
                device="cpu",
                on_epoch_end=logs.append,
            )
        else:
            joint_train_backbone_and_exits(
                _build_wrapper(),
                _toy_loader(n=16),
                epochs=1,
                device="cpu",
                on_epoch_end=logs.append,
            )
    assert logs[0].val_loss is None and logs[0].val_accuracy is None
    assert not [w for w in captured if "val_loader" in str(w.message)]


def test_stage2_validation_does_not_update_backbone():
    """The per-epoch validation forward pass must not mutate the frozen backbone:
    BN running stats and weights stay put even with a val_loader and >1 epoch."""
    wrapper = _build_wrapper()
    bb_param = wrapper.backbone.stage1[0].weight.detach().clone()
    bn_mean = wrapper.backbone.stage1[1].running_mean.detach().clone()
    bn_var = wrapper.backbone.stage1[1].running_var.detach().clone()

    stage2_train_exits(
        wrapper,
        _toy_loader(n=16),
        val_loader=_toy_loader(n=8),
        epochs=2,
        lr=1e-2,
        device="cpu",
        on_epoch_end=lambda _: None,
    )

    assert torch.equal(bb_param, wrapper.backbone.stage1[0].weight.detach())
    assert torch.equal(bn_mean, wrapper.backbone.stage1[1].running_mean.detach())
    assert torch.equal(bn_var, wrapper.backbone.stage1[1].running_var.detach())


def test_validation_restores_training_modes():
    """Passing a val_loader must not change the returned model's train/eval mode:
    the result is identical to the no-val run (validate snapshots + restores)."""
    no_val = _build_wrapper()
    stage2_train_exits(
        no_val, _toy_loader(n=8), epochs=1, device="cpu", on_epoch_end=lambda _: None
    )
    with_val = _build_wrapper()
    stage2_train_exits(
        with_val,
        _toy_loader(n=8),
        val_loader=_toy_loader(n=8),
        epochs=1,
        device="cpu",
        on_epoch_end=lambda _: None,
    )
    modes_no_val = {n: m.training for n, m in no_val.named_modules()}
    modes_with_val = {n: m.training for n, m in with_val.named_modules()}
    assert modes_no_val == modes_with_val


def test_stage2_val_metrics_match_recomputation():
    """The logged val metrics are the real computed numbers from the val loader,
    not a placeholder — they match an independent recomputation on the same model."""
    import math

    from earlyon.training.two_stage_trainer import _validate_wrapper

    torch.manual_seed(0)
    wrapper = _build_wrapper()
    val = DataLoader(
        TensorDataset(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))),
        batch_size=8,
        shuffle=False,
    )
    logs: list = []
    stage2_train_exits(
        wrapper, _toy_loader(n=16), val_loader=val, epochs=1, device="cpu", on_epoch_end=logs.append
    )
    exp_loss, exp_acc = _validate_wrapper(wrapper, val, "cpu")
    assert math.isclose(logs[-1].val_loss, exp_loss, rel_tol=1e-6)
    assert math.isclose(logs[-1].val_accuracy, exp_acc, rel_tol=1e-6)


def test_empty_val_loader_raises():
    """An empty val_loader is a caller error; validation must fail fast rather
    than report a misleading (0.0, 0.0)."""
    wrapper = _build_wrapper()
    empty = DataLoader(
        TensorDataset(torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long)),
        batch_size=8,
    )
    with pytest.raises(ValueError, match="no batches"):
        stage2_train_exits(
            wrapper,
            _toy_loader(n=8),
            val_loader=empty,
            epochs=1,
            device="cpu",
            on_epoch_end=lambda _: None,
        )


def test_stage2_does_not_update_backbone():
    """Critical: stage 2 must leave backbone params and BN running stats unchanged."""
    wrapper = _build_wrapper()
    # snapshot a backbone parameter and BN running mean
    bb_param = wrapper.backbone.stage1[0].weight.detach().clone()
    bn_running_mean = wrapper.backbone.stage1[1].running_mean.detach().clone()

    loader = _toy_loader()
    stage2_train_exits(
        wrapper, loader, epochs=1, lr=1e-2, device="cpu", on_epoch_end=lambda _: None
    )

    new_param = wrapper.backbone.stage1[0].weight.detach()
    new_bn = wrapper.backbone.stage1[1].running_mean.detach()
    assert torch.equal(bb_param, new_param), "backbone weights changed during stage 2"
    assert torch.equal(bn_running_mean, new_bn), "BN running_mean changed during stage 2"


def test_stage2_updates_exit_heads():
    wrapper = _build_wrapper()
    head_param_before = next(wrapper.exit_heads["e0"].parameters()).detach().clone()
    loader = _toy_loader()
    stage2_train_exits(
        wrapper, loader, epochs=1, lr=1e-2, device="cpu", on_epoch_end=lambda _: None
    )
    head_param_after = next(wrapper.exit_heads["e0"].parameters()).detach()
    assert not torch.equal(head_param_before, head_param_after), "exit head did not update"


def test_joint_trainer_updates_backbone_and_exits():
    """Joint training updates backbone parameters AND exit-head parameters
    in the same loop. Counterpart to two-stage, exposed for users who want
    end-to-end gradient flow."""
    wrapper = _build_wrapper()
    bb_param_before = wrapper.backbone.stage1[0].weight.detach().clone()
    head_param_before = next(wrapper.exit_heads["e0"].parameters()).detach().clone()

    loader = _toy_loader()
    joint_train_backbone_and_exits(
        wrapper, loader, epochs=1, lr=1e-2, device="cpu", on_epoch_end=lambda _: None
    )

    bb_param_after = wrapper.backbone.stage1[0].weight.detach()
    head_param_after = next(wrapper.exit_heads["e0"].parameters()).detach()
    assert not torch.equal(bb_param_before, bb_param_after), "backbone did not update"
    assert not torch.equal(head_param_before, head_param_after), "exit head did not update"


def test_joint_trainer_does_not_freeze_batchnorm():
    """Joint training keeps BN in train mode so running stats update — unlike
    stage 2 which deliberately freezes them."""
    wrapper = _build_wrapper()
    bn_running_mean_before = wrapper.backbone.stage1[1].running_mean.detach().clone()
    loader = _toy_loader()
    joint_train_backbone_and_exits(
        wrapper, loader, epochs=1, lr=1e-2, device="cpu", on_epoch_end=lambda _: None
    )
    bn_running_mean_after = wrapper.backbone.stage1[1].running_mean.detach()
    assert not torch.equal(
        bn_running_mean_before, bn_running_mean_after
    ), "BN running_mean did not update; backbone was unexpectedly frozen"


def test_joint_trainer_log_reports_per_exit_accuracy():
    wrapper = _build_wrapper()
    loader = _toy_loader()
    captured: list = []

    joint_train_backbone_and_exits(
        wrapper, loader, epochs=1, lr=1e-2, device="cpu", on_epoch_end=captured.append
    )
    assert len(captured) == 1
    log = captured[0]
    assert log.per_exit_accuracy is not None
    n_exits = len(wrapper.config.exit_points)
    assert set(log.per_exit_accuracy.keys()) == {f"exit_{i}" for i in range(n_exits)} | {"final"}


def test_stage2_log_carries_per_exit_accuracy():
    """stage 2 must populate per_exit_accuracy and report a mean across all
    outputs (not just exit_0)."""
    wrapper = _build_wrapper()
    loader = _toy_loader()
    captured: list = []

    def cb(log):
        captured.append(log)

    stage2_train_exits(wrapper, loader, epochs=1, lr=1e-2, device="cpu", on_epoch_end=cb)

    assert len(captured) == 1
    log = captured[0]
    assert log.per_exit_accuracy is not None
    n_exits = len(wrapper.config.exit_points)
    # one entry per exit head plus one for the final classifier
    assert set(log.per_exit_accuracy.keys()) == {f"exit_{i}" for i in range(n_exits)} | {"final"}
    # log.accuracy is the mean across outputs
    mean = sum(log.per_exit_accuracy.values()) / len(log.per_exit_accuracy)
    assert abs(log.accuracy - mean) < 1e-9
