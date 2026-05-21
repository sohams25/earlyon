import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from earlyon.core.exit_head import EarlyExitHead
from earlyon.core.types import EarlyExitConfig, ExitPoint
from earlyon.core.wrappers import EarlyExitWrapper
from earlyon.training import (
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
        backbone="tiny", num_classes=10, exit_points=exits,
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
    out = stage1_train_backbone(bb, loader, epochs=1, lr=0.01, device="cpu", on_epoch_end=lambda _: None)
    assert out is bb


def test_stage2_does_not_update_backbone():
    """Critical: stage 2 must leave backbone params and BN running stats unchanged."""
    wrapper = _build_wrapper()
    # snapshot a backbone parameter and BN running mean
    bb_param = wrapper.backbone.stage1[0].weight.detach().clone()
    bn_running_mean = wrapper.backbone.stage1[1].running_mean.detach().clone()

    loader = _toy_loader()
    stage2_train_exits(wrapper, loader, epochs=1, lr=1e-2, device="cpu", on_epoch_end=lambda _: None)

    new_param = wrapper.backbone.stage1[0].weight.detach()
    new_bn = wrapper.backbone.stage1[1].running_mean.detach()
    assert torch.equal(bb_param, new_param), "backbone weights changed during stage 2"
    assert torch.equal(bn_running_mean, new_bn), "BN running_mean changed during stage 2"


def test_stage2_updates_exit_heads():
    wrapper = _build_wrapper()
    head_param_before = next(wrapper.exit_heads["e0"].parameters()).detach().clone()
    loader = _toy_loader()
    stage2_train_exits(wrapper, loader, epochs=1, lr=1e-2, device="cpu", on_epoch_end=lambda _: None)
    head_param_after = next(wrapper.exit_heads["e0"].parameters()).detach()
    assert not torch.equal(head_param_before, head_param_after), "exit head did not update"
