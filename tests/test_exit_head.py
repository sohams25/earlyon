import pytest
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


def test_4d_conv_features_still_supported():
    """The CNN path (4D spatial features) must be unchanged."""
    head = EarlyExitHead(in_channels=512, num_classes=10)
    assert head(torch.randn(2, 512, 7, 7)).shape == (2, 10)


@pytest.mark.parametrize("pool", ["cls", "mean"])
def test_3d_token_features(pool):
    """Transformer token features (B, N, D) classify on D regardless of pooling."""
    head = EarlyExitHead(in_channels=768, num_classes=10, pool_tokens=pool)
    assert head(torch.randn(2, 197, 768)).shape == (2, 10)


def test_cls_and_mean_pooling_differ():
    """cls (token 0) and mean pooling produce different logits on non-uniform
    token features — proving the dispatch actually differs."""
    torch.manual_seed(0)
    x = torch.randn(2, 197, 768)
    cls_head = EarlyExitHead(768, 10, pool_tokens="cls")
    mean_head = EarlyExitHead(768, 10, pool_tokens="mean")
    mean_head.load_state_dict(cls_head.state_dict())  # identical weights
    assert not torch.allclose(cls_head(x), mean_head(x))


def test_2d_pooled_features_pass_through():
    """Already-pooled (B, D) vectors are accepted unchanged."""
    head = EarlyExitHead(in_channels=64, num_classes=5)
    assert head(torch.randn(3, 64)).shape == (3, 5)


def test_invalid_rank_raises():
    head = EarlyExitHead(in_channels=16, num_classes=4)
    with pytest.raises(ValueError, match="2D, 3D, or 4D"):
        head(torch.randn(1, 16, 4, 4, 4))


def test_invalid_pool_tokens_raises():
    with pytest.raises(ValueError, match="pool_tokens"):
        EarlyExitHead(in_channels=16, num_classes=4, pool_tokens="sum")
