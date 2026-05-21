import pytest
import torch


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def seed_everything():
    torch.manual_seed(42)
