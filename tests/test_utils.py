"""Tests for earlyon.utils — dataset loaders and the build_model factory map.

cifar10_loaders is exercised with torchvision.datasets.CIFAR10 monkeypatched to a
tiny in-memory dataset so the tests need no network access or real download.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from earlyon.utils import build_model, cifar10_loaders


class _FakeCIFAR10:
    """Minimal stand-in for torchvision.datasets.CIFAR10 — fixed-size, returns
    already-tensorised samples and ignores the transform/download args."""

    def __init__(self, root=None, train=True, download=False, transform=None):
        self.n = 100

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return torch.zeros(3, 8, 8), index % 10


@pytest.fixture
def _patched_cifar(monkeypatch):
    import torchvision

    monkeypatch.setattr(torchvision.datasets, "CIFAR10", _FakeCIFAR10)


def test_cifar10_loaders_returns_three_dataloaders(_patched_cifar):
    train, val, test = cifar10_loaders(image_size=8, num_workers=0)
    assert isinstance(train, DataLoader)
    assert isinstance(val, DataLoader)
    assert isinstance(test, DataLoader)


def test_cifar10_loaders_val_and_test_are_batch_size_1(_patched_cifar):
    """The routing contract: evaluate/benchmark require batch_size=1, so the val
    and test loaders must be batch-1 regardless of the train batch size."""
    _train, val, test = cifar10_loaders(batch_size=64, image_size=8, num_workers=0)
    assert val.batch_size == 1
    assert test.batch_size == 1


def test_cifar10_loaders_honors_val_batch_size(_patched_cifar):
    """val_batch_size lets training-time validation be batched while the default
    (1) preserves the single-sample contract calibrate/analyze rely on."""
    _train, val_default, _t = cifar10_loaders(image_size=8, num_workers=0)
    assert val_default.batch_size == 1
    _train, val_batched, _t = cifar10_loaders(image_size=8, num_workers=0, val_batch_size=16)
    assert val_batched.batch_size == 16


def test_cifar10_loaders_val_split_partitions_train(_patched_cifar):
    """val_split carves a disjoint slice off the training set; train+val sizes
    must sum to the full training set and not overlap."""
    train, val, _test = cifar10_loaders(val_split=0.1, image_size=8, num_workers=0)
    assert len(val.dataset) == 10  # int(100 * 0.1)
    assert len(train.dataset) == 90
    # Subset.indices expose the partition; they must be disjoint.
    train_idx = set(train.dataset.indices)
    val_idx = set(val.dataset.indices)
    assert train_idx.isdisjoint(val_idx)


def test_build_model_rejects_unknown_backbone():
    with pytest.raises(ValueError, match="unknown backbone"):
        build_model("not_a_real_backbone", num_classes=10, pretrained=False)


def test_build_model_known_backbone_constructs_wrapper():
    model = build_model("resnet18", num_classes=10, pretrained=False)
    assert model.config.backbone == "resnet18"
    assert model.config.num_classes == 10


def test_build_model_rejects_custom_backbone():
    """custom_ee models can't be reconstructed from a string — build_model must
    raise a clear NotImplementedError, not the generic 'unknown backbone'."""
    with pytest.raises(NotImplementedError, match="custom_ee"):
        build_model("custom", num_classes=10)


def test_build_model_constructs_vit():
    """vit_b_16 is a recognized FACTORIES backbone (the CLI relies on this)."""
    model = build_model("vit_b_16", num_classes=10, pretrained=False)
    assert model.config.backbone == "vit_b_16"
    assert len(model.config.exit_points) == 2
