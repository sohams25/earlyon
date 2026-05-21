"""Train a ResNet50 + early exits on CIFAR-10.

This is the end-to-end recipe. Run with:

    python examples/01_train_resnet50_cifar10.py

Stage 1 (~1-2h on RTX 3060): standard ResNet50 training, no exits involved.
Stage 2 (~10-20min): freezes the backbone and trains the 3 lightweight
exit heads with weighted multi-exit cross-entropy.

For a faster run, drop --epochs to 10 / 5.
"""

import argparse

import torch

from earlyon.core.thresholds import calibrate_thresholds
from earlyon.training import stage1_train_backbone, stage2_train_exits
from earlyon.utils import build_model, cifar10_loaders, save_wrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-stage1", type=int, default=90)
    parser.add_argument("--epochs-stage2", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="resnet50_ee_cifar10.pth")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_loader, val_loader, _ = cifar10_loaders(batch_size=args.batch_size)
    model = build_model("resnet50", num_classes=10, pretrained=True)

    print("stage 1: training backbone")
    stage1_train_backbone(
        model.backbone, train_loader, epochs=args.epochs_stage1,
        lr=0.01, device=device,  # lower lr because we start from pretrained
    )

    print("stage 2: training exit heads")
    stage2_train_exits(
        model, train_loader, epochs=args.epochs_stage2, lr=1e-3, device=device,
    )

    print("calibrating thresholds on validation set")
    result = calibrate_thresholds(
        model, val_loader, target_accuracy_drop=0.01, device=device,
    )
    print(f"thresholds: {result.thresholds}")
    print(f"baseline acc: {result.baseline_accuracy:.4f}  "
          f"final acc: {result.final_accuracy:.4f}  "
          f"avg compute: {result.avg_computation_used:.4f}")

    save_wrapper(model, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
