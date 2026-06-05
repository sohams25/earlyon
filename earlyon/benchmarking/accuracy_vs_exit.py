"""Per-exit accuracy breakdown and exit-distribution analysis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from earlyon.core.types import Batch, exit_label
from earlyon.core.wrappers import EarlyExitWrapper


@dataclass
class AccuracyReport:
    overall_accuracy: float
    avg_computation_used: float
    exit_distribution: dict[str, float]
    per_exit_accuracy: dict[str, float]
    per_class_exit_distribution: dict[str, dict[str, float]]


def evaluate(
    model: EarlyExitWrapper,
    loader: DataLoader[Batch],
    device: str = "cpu",
    class_names: list[str] | None = None,
) -> AccuracyReport:
    """Run inference (batch=1) and report accuracy, exit distribution, per-class exits.

    The loader is iterated and each sample is processed individually to obey
    the v0.1 batch_size=1 routing constraint.
    """
    model = model.to(device).eval()
    correct = 0
    total = 0
    comp_used_sum = 0.0
    exit_counts: dict[str, int] = defaultdict(int)
    exit_correct: dict[str, int] = defaultdict(int)
    class_exits: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            for img, tgt in zip(images, targets):
                result = model(img.unsqueeze(0), mode="inference")
                pred = int(result.prediction.argmax(dim=-1).item())
                tgt_i = int(tgt.item())
                ok = pred == tgt_i
                correct += int(ok)
                total += 1
                comp_used_sum += result.computation_used
                key = exit_label(result.exit_taken)
                exit_counts[key] += 1
                if ok:
                    exit_correct[key] += 1
                class_key = (
                    class_names[tgt_i] if class_names and tgt_i < len(class_names) else str(tgt_i)
                )
                class_exits[class_key][key] += 1

    per_exit_acc = {
        k: (exit_correct[k] / exit_counts[k]) if exit_counts[k] else 0.0 for k in exit_counts
    }
    per_class_dist = {
        cls: {k: v / sum(counts.values()) for k, v in counts.items()}
        for cls, counts in class_exits.items()
    }
    return AccuracyReport(
        overall_accuracy=correct / max(total, 1),
        avg_computation_used=comp_used_sum / max(total, 1),
        exit_distribution={k: v / max(total, 1) for k, v in exit_counts.items()},
        per_exit_accuracy=per_exit_acc,
        per_class_exit_distribution=per_class_dist,
    )
