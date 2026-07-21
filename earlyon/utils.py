"""Small shared helpers: dataset loaders, checkpoint save/load.

Checkpoint contract (format_version 2)
--------------------------------------
``save_wrapper`` writes a dict with:

* ``format_version`` — integer schema version (currently 2).
* ``earlyon_version`` — the library version that wrote the file.
* ``config`` — everything needed to reconstruct routing behavior when the
  factory is known: backbone id, num_classes, routing policy, both threshold
  lists, ``enabled_exits``, per-head ``temperatures``, loss weights, and the
  exit points (name / layer_name / in_channels).
* ``state_dict`` — the wrapper's weights.

``load_wrapper`` reads v2 files directly and migrates unversioned (v1) files:
the legacy scalar ``temperature`` is broadcast to every head, and the legacy
"disabled" threshold sentinels (confidence ``1.0`` / entropy ``0.0``, for the
active policy only) become explicit ``enabled_exits=False`` entries, with a
``UserWarning`` describing the migration.

Arbitrary ``custom_ee`` backbones cannot be reconstructed from a string; pass
``factory=`` (a zero-argument callable returning a structurally identical
wrapper) to load them.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Callable, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from earlyon import __version__
from earlyon.core.types import FINAL_HEAD, Batch
from earlyon.core.wrappers import EarlyExitWrapper
from earlyon.models import (
    cifar_resnet_ee,
    efficientnet_b0_ee,
    mobilenetv2_ee,
    resnet18_ee,
    resnet50_ee,
    vit_b_16_ee,
)

_CIFAR_PREFIX = "cifar_resnet"

CHECKPOINT_FORMAT_VERSION = 2

FACTORIES = {
    "resnet18": resnet18_ee,
    "resnet50": resnet50_ee,
    "mobilenetv2": mobilenetv2_ee,
    "efficientnet_b0": efficientnet_b0_ee,
    "vit_b_16": vit_b_16_ee,
}


def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> EarlyExitWrapper:
    # custom_ee models can't be reconstructed from a string — fail loudly rather
    # than with a misleading "unknown backbone".
    if backbone == "custom":
        raise NotImplementedError(
            "custom_ee models are not round-trippable via build_model/load_wrapper: "
            "rebuild the backbone, re-wrap with custom_ee(...), then load the saved "
            "state_dict yourself"
        )
    # cifar_resnet_ee records its depth in the backbone string (e.g.
    # "cifar_resnet20"); reconstruct the architecture from that so save/load
    # round-trips. Without this, any cifar_resnet checkpoint is unloadable.
    if backbone.startswith(_CIFAR_PREFIX):
        suffix = backbone[len(_CIFAR_PREFIX) :]
        if not suffix.isdigit():
            raise ValueError(
                f"malformed cifar_resnet backbone {backbone!r}; expected e.g. 'cifar_resnet20'"
            )
        return cifar_resnet_ee(num_classes=num_classes, depth=int(suffix))
    if backbone not in FACTORIES:
        raise ValueError(
            f"unknown backbone {backbone!r}; choose from {list(FACTORIES)} or 'cifar_resnet<depth>'"
        )
    return FACTORIES[backbone](num_classes=num_classes, pretrained=pretrained)


def save_wrapper(model: EarlyExitWrapper, path: str | Path) -> None:
    """Save state_dict + a versioned config (format_version 2).

    The config records everything needed to reconstruct routing behavior when
    the model factory is known — see the module docstring for the contract.
    """
    if model.config.backbone == "custom":
        warnings.warn(
            "saving a custom_ee model: load_wrapper cannot rebuild an arbitrary "
            "backbone from a string. To restore it, rebuild the backbone and pass "
            "factory=lambda: custom_ee(...) to load_wrapper.",
            UserWarning,
            stacklevel=2,
        )
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "earlyon_version": __version__,
        "state_dict": model.state_dict(),
        "config": {
            "backbone": model.config.backbone,
            "num_classes": model.config.num_classes,
            "routing_policy": model.config.routing_policy,
            "confidence_thresholds": list(model.config.confidence_thresholds),
            "entropy_thresholds": list(model.config.entropy_thresholds),
            "enabled_exits": list(model.config.enabled_exits),
            "temperatures": dict(model.config.temperatures),
            "loss_weights": list(model.config.loss_weights),
            "exit_points": [
                {"name": ep.name, "layer_name": ep.layer_name, "in_channels": ep.in_channels}
                for ep in model.config.exit_points
            ],
        },
    }
    torch.save(payload, Path(path))


def _migrate_v1_config(cfg: dict[str, Any], model: EarlyExitWrapper, name: str) -> dict[str, Any]:
    """Translate an unversioned (v1) checkpoint config into v2 shape.

    Deterministic rules:

    * the legacy scalar ``temperature`` is broadcast to every head (that is
      exactly what v1 routing did); a non-finite/non-positive scalar falls
      back to 1.0 with a warning instead of failing the load, matching the
      v1 runtime guard.
    * the legacy "disabled" sentinels — confidence threshold ``1.0`` /
      entropy threshold ``0.0`` on the *active* policy — become explicit
      ``enabled_exits=False``. Under v1 semantics such an exit could still
      fire on a numerically saturated softmax; the documented intent was
      "disabled", which the explicit flag now honors exactly.
    """
    migrated = dict(cfg)
    n_exits = len(model.config.exit_points)

    temperature = float(cfg.get("temperature", 1.0))
    if not math.isfinite(temperature) or temperature <= 0.0:
        warnings.warn(
            f"checkpoint {name}: legacy temperature {temperature} is invalid; "
            "falling back to 1.0 (uncalibrated)",
            UserWarning,
            stacklevel=3,
        )
        temperature = 1.0
    head_names = [ep.name for ep in model.config.exit_points] + [FINAL_HEAD]
    migrated["temperatures"] = {h: temperature for h in head_names}

    policy = cfg.get("routing_policy", "confidence")
    if policy == "confidence":
        active = [float(t) for t in cfg.get("confidence_thresholds", [])]
        sentinel = 1.0
    else:
        active = [float(t) for t in cfg.get("entropy_thresholds", [0.0] * n_exits)]
        sentinel = 0.0
    enabled = [t != sentinel for t in active]
    migrated["enabled_exits"] = enabled
    if not all(enabled):
        disabled_idx = [i for i, e in enumerate(enabled) if not e]
        warnings.warn(
            f"checkpoint {name}: migrated v1 sentinel threshold(s) at exit(s) "
            f"{disabled_idx} to explicit enabled_exits=False. Under v1 these "
            "exits could still fire on a saturated softmax; they are now "
            "strictly disabled.",
            UserWarning,
            stacklevel=3,
        )
    return migrated


def load_wrapper(
    path: str | Path,
    pretrained_backbone: bool = False,
    factory: Callable[[], EarlyExitWrapper] | None = None,
) -> EarlyExitWrapper:
    """Load a checkpoint written by :func:`save_wrapper`.

    ``factory`` — required for ``custom_ee`` models — must return a wrapper
    structurally identical to the saved one; the checkpoint's routing config
    and weights are then applied to it. Built-in backbones are reconstructed
    automatically via :func:`build_model`.
    """
    # weights_only=True prevents pickle code-exec from malicious .pth files
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    name = Path(path).name
    if "config" not in payload or "state_dict" not in payload:
        raise ValueError(f"checkpoint {name}: missing 'config'/'state_dict'; not an earlyon file")
    version = int(payload.get("format_version", 1))
    if version > CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"checkpoint {name}: format_version {version} is newer than this "
            f"earlyon supports ({CHECKPOINT_FORMAT_VERSION}); upgrade earlyon"
        )
    cfg = payload["config"]

    if factory is not None:
        model = factory()
    elif cfg["backbone"] == "custom":
        raise ValueError(
            f"checkpoint {name}: backbone is 'custom', which cannot be rebuilt "
            "from a string. Reconstruct the backbone and pass "
            "factory=lambda: custom_ee(backbone, exit_layers=[...], ...) with the "
            "same structure it was saved with."
        )
    else:
        model = build_model(cfg["backbone"], cfg["num_classes"], pretrained=pretrained_backbone)

    if version < 2:
        cfg = _migrate_v1_config(cfg, model, name)

    # Validate config shape BEFORE mutating model.config. A checkpoint saved
    # against a different exit count would otherwise leave model.config in a
    # half-updated state if load_state_dict raised later.
    n_exits = len(model.config.exit_points)
    thr_len = len(cfg["confidence_thresholds"])
    weight_len = len(cfg["loss_weights"])
    if thr_len != n_exits:
        raise ValueError(
            f"checkpoint {name}: confidence_thresholds has "
            f"length {thr_len}, but backbone {cfg['backbone']!r} has "
            f"{n_exits} exit points"
        )
    if weight_len != n_exits + 1:
        raise ValueError(
            f"checkpoint {name}: loss_weights has length "
            f"{weight_len}, expected {n_exits + 1} (one per exit plus final)"
        )
    # entropy_thresholds is optional in v1 files; validate when present.
    if "entropy_thresholds" in cfg and len(cfg["entropy_thresholds"]) != n_exits:
        raise ValueError(
            f"checkpoint {name}: entropy_thresholds has length "
            f"{len(cfg['entropy_thresholds'])}, but backbone {cfg['backbone']!r} "
            f"has {n_exits} exit points"
        )
    policy = cfg.get("routing_policy", model.config.routing_policy)
    if policy not in {"confidence", "entropy"}:
        raise ValueError(
            f"checkpoint {name}: unsupported routing_policy {policy!r} "
            "(allowed: 'confidence', 'entropy')"
        )
    if policy == "entropy" and "entropy_thresholds" not in cfg:
        raise ValueError(
            f"checkpoint {name}: routing_policy='entropy' but no "
            "entropy_thresholds were persisted — the model would silently route "
            "on uncalibrated defaults"
        )
    # v2 files carry the exit points; cross-check them against the rebuilt
    # model so a checkpoint from a different factory revision fails loudly.
    if "exit_points" in cfg:
        saved = [(p["name"], p["layer_name"], int(p["in_channels"])) for p in cfg["exit_points"]]
        built = [(ep.name, ep.layer_name, ep.in_channels) for ep in model.config.exit_points]
        if saved != built:
            raise ValueError(
                f"checkpoint {name}: saved exit points {saved} do not match the "
                f"reconstructed model's exit points {built}; the factory or "
                "earlyon version that wrote this file placed exits differently"
            )

    model.config.confidence_thresholds = [float(t) for t in cfg["confidence_thresholds"]]
    model.config.loss_weights = [float(w) for w in cfg["loss_weights"]]
    model.config.routing_policy = policy
    if "entropy_thresholds" in cfg:
        model.config.entropy_thresholds = [float(t) for t in cfg["entropy_thresholds"]]
    model.config.enabled_exits = [bool(e) for e in cfg["enabled_exits"]]
    model.config.temperatures = {k: float(v) for k, v in cfg["temperatures"].items()}
    model.config.validate()
    model.load_state_dict(payload["state_dict"])
    return model


def cifar10_loaders(
    root: str = "./data",
    batch_size: int = 128,
    image_size: int = 224,
    num_workers: int = 2,
    val_split: float = 0.1,
    val_batch_size: int = 1,
) -> Tuple[DataLoader[Batch], DataLoader[Batch], DataLoader[Batch]]:
    """Return (train, val, test) DataLoaders for CIFAR-10.

    Images are upsampled to ``image_size`` so torchvision ImageNet-pretrained
    backbones work without modification. For honest benchmarking on CIFAR
    natively you'd want a CIFAR-specific ResNet variant; v0.1 prioritizes
    the pip-install story.

    ``val_batch_size`` defaults to 1 (single-sample routing, as calibrate/analyze
    require); pass a larger value for fast batched validation during training.
    The test loader is always batch_size=1 for routing evaluation.
    """
    import torchvision
    from torchvision import transforms

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_full = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_tf
    )
    test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=eval_tf)
    n_val = int(len(train_full) * val_split)
    indices = list(range(len(train_full)))
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    val_full = torchvision.datasets.CIFAR10(
        root=root, train=True, download=False, transform=eval_tf
    )
    train_loader = DataLoader(
        Subset(train_full, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_full, val_idx),
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
