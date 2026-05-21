import torch

from earlyon.core.exit_head import EarlyExitHead


def test_output_shape():
    head = EarlyExitHead(in_channels=32, num_classes=10)
    feats = torch.randn(4, 32, 8, 8)
    logits = head(feats)
    assert logits.shape == (4, 10)


def test_parameter_count_is_small():
    # 64 channels -> 10 classes -> ~10k params (the head must be lightweight)
    head = EarlyExitHead(in_channels=64, num_classes=10)
    total = sum(p.numel() for p in head.parameters())
    assert total < 20_000, f"head too large: {total} params"
