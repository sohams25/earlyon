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


def test_save_wrapper_warns_for_custom_backbone(tmp_path):
    """custom_ee artifacts cannot be rebuilt by load_wrapper; the user must
    hear that at SAVE time, not after training when the load fails."""
    import warnings

    import torch
    from torch import nn

    from earlyon.models import custom_ee
    from earlyon.utils import save_wrapper

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Conv2d(3, 8, 3, padding=1)
            self.head = nn.Linear(8, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feats = self.body(x)
            return self.head(feats.mean(dim=(2, 3)))

    model = custom_ee(Tiny(), ["body"], num_classes=10, input_shape=(1, 3, 16, 16))
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        save_wrapper(model, tmp_path / "m.pth")
    messages = [str(w.message) for w in captured if issubclass(w.category, UserWarning)]
    assert any("load_wrapper" in m for m in messages), messages


# ---------------- checkpoint format v2 ----------------


def test_checkpoint_v2_payload_shape(tmp_path):
    """v2 files carry format_version, library version, and the full routing
    config including exit points — the documented public contract."""
    import torch

    from earlyon import __version__
    from earlyon.utils import build_model, save_wrapper

    model = build_model("resnet18", num_classes=10, pretrained=False)
    path = tmp_path / "v2.pth"
    save_wrapper(model, path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["format_version"] == 2
    assert payload["earlyon_version"] == __version__
    cfg = payload["config"]
    assert cfg["enabled_exits"] == [True, True]
    assert set(cfg["temperatures"]) == {"e0", "e1", "final"}
    assert [p["layer_name"] for p in cfg["exit_points"]] == ["layer2", "layer3"]


def test_v1_checkpoint_migrates_scalar_temperature_and_sentinels(tmp_path):
    """Unversioned (v1) checkpoints must load: the scalar temperature is
    broadcast per head, and the legacy 'disabled' sentinel (confidence 1.0 on
    the active policy) becomes an explicit enabled_exits=False with a warning."""
    import warnings as _warnings

    import torch

    from earlyon.utils import build_model, load_wrapper

    model = build_model("resnet18", num_classes=10, pretrained=False)
    path = tmp_path / "v1.pth"
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "backbone": "resnet18",
            "num_classes": 10,
            "routing_policy": "confidence",
            "confidence_thresholds": [0.7, 1.0],  # 1.0 was the v1 disabled sentinel
            "entropy_thresholds": [0.5, 0.5],
            "loss_weights": [0.2, 0.3, 0.5],
            "temperature": 1.7,
        },
    }
    torch.save(payload, path)

    with _warnings.catch_warnings(record=True) as captured:
        _warnings.simplefilter("always")
        loaded = load_wrapper(path)
    assert loaded.config.temperatures == {"e0": 1.7, "e1": 1.7, "final": 1.7}
    assert loaded.config.enabled_exits == [True, False]
    messages = [str(w.message) for w in captured if issubclass(w.category, UserWarning)]
    assert any("enabled_exits" in m for m in messages), messages


def test_v1_checkpoint_entropy_sentinel_migrates(tmp_path):
    """For an entropy-routed v1 checkpoint the disabled sentinel is 0.0."""
    import torch

    from earlyon.utils import build_model, load_wrapper

    model = build_model("resnet18", num_classes=10, pretrained=False)
    path = tmp_path / "v1e.pth"
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "backbone": "resnet18",
            "num_classes": 10,
            "routing_policy": "entropy",
            "confidence_thresholds": [0.85, 0.80],
            "entropy_thresholds": [0.4, 0.0],  # 0.0 was the v1 entropy disabled sentinel
            "loss_weights": [0.2, 0.3, 0.5],
            "temperature": 1.0,
        },
    }
    torch.save(payload, path)
    loaded = load_wrapper(path)
    assert loaded.config.routing_policy == "entropy"
    assert loaded.config.enabled_exits == [True, False]


def test_v1_checkpoint_invalid_temperature_falls_back(tmp_path):
    """A corrupt v1 temperature (NaN/negative) must not poison the load; it
    falls back to 1.0 with a warning, matching the v1 runtime guard."""
    import warnings as _warnings

    import torch

    from earlyon.utils import build_model, load_wrapper

    model = build_model("resnet18", num_classes=10, pretrained=False)
    path = tmp_path / "v1bad.pth"
    payload = {
        "state_dict": model.state_dict(),
        "config": {
            "backbone": "resnet18",
            "num_classes": 10,
            "confidence_thresholds": [0.7, 0.8],
            "loss_weights": [0.2, 0.3, 0.5],
            "temperature": float("nan"),
        },
    }
    torch.save(payload, path)
    with _warnings.catch_warnings(record=True) as captured:
        _warnings.simplefilter("always")
        loaded = load_wrapper(path)
    assert loaded.config.temperatures == {"e0": 1.0, "e1": 1.0, "final": 1.0}
    assert any("temperature" in str(w.message) for w in captured)


def test_checkpoint_from_newer_format_rejected(tmp_path):
    import pytest
    import torch

    from earlyon.utils import build_model, load_wrapper, save_wrapper

    model = build_model("resnet18", num_classes=10, pretrained=False)
    path = tmp_path / "future.pth"
    save_wrapper(model, path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["format_version"] = 99
    torch.save(payload, path)
    with pytest.raises(ValueError, match="format_version"):
        load_wrapper(path)


def test_malformed_checkpoint_rejected(tmp_path):
    import pytest
    import torch

    from earlyon.utils import load_wrapper

    path = tmp_path / "junk.pth"
    torch.save({"weights": {}}, path)
    with pytest.raises(ValueError, match="not an earlyon file"):
        load_wrapper(path)


def test_checkpoint_exit_point_mismatch_rejected(tmp_path):
    """A v2 checkpoint whose recorded exit points disagree with the rebuilt
    factory's placement must fail loudly instead of loading mis-wired routing."""
    import pytest
    import torch

    from earlyon.utils import build_model, load_wrapper, save_wrapper

    model = build_model("resnet18", num_classes=10, pretrained=False)
    path = tmp_path / "moved.pth"
    save_wrapper(model, path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["config"]["exit_points"][0]["layer_name"] = "layer1"  # tampered placement
    torch.save(payload, path)
    with pytest.raises(ValueError, match="exit points"):
        load_wrapper(path)


def test_custom_checkpoint_loads_via_factory(tmp_path):
    """custom_ee models round-trip when the caller supplies the factory."""
    import torch

    from earlyon.models import custom_ee
    from earlyon.utils import load_wrapper, save_wrapper
    from tests.fixtures.tiny_models import TinyBackbone

    torch.manual_seed(0)

    def factory():
        return custom_ee(
            TinyBackbone(num_classes=10),
            exit_layers=["stage1", "stage2"],
            num_classes=10,
            input_shape=(1, 3, 32, 32),
        )

    model = factory()
    model.config.confidence_thresholds = [0.42, 0.9]
    model.config.enabled_exits = [True, False]
    with torch.no_grad():
        next(iter(model.exit_heads["e0"].parameters())).fill_(0.5)

    path = tmp_path / "custom.pth"
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")  # the save-time custom_ee warning
        save_wrapper(model, path)

    loaded = load_wrapper(path, factory=factory)
    assert loaded.config.confidence_thresholds == [0.42, 0.9]
    assert loaded.config.enabled_exits == [True, False]
    p = next(iter(loaded.exit_heads["e0"].parameters()))
    assert torch.allclose(p, torch.full_like(p, 0.5))
