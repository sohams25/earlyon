"""GPU-only tests (@pytest.mark.gpu).

CI runs with ``-m "not gpu"`` so these are skipped there; they exercise the
CUDA paths — device placement, the ``torch.cuda.synchronize`` branch in the
benchmark harness, and the stage-2 BatchNorm-freeze invariant on-device — that
the CPU suite structurally cannot reach. Each is additionally guarded with
``skipif`` so running ``-m gpu`` on a CPU-only box skips rather than errors.
"""

import pytest
import torch

from earlyon.benchmarking import benchmark_wrapper
from earlyon.core.exit_head import EarlyExitHead
from earlyon.core.types import EarlyExitConfig, ExitPoint
from earlyon.core.wrappers import EarlyExitWrapper
from earlyon.training import stage2_train_exits
from tests.fixtures.tiny_models import STAGE_CHANNELS, TinyBackbone

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device"),
]


def _build(thresholds=(1.0, 1.0)):
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
        confidence_thresholds=list(thresholds),
    )
    return EarlyExitWrapper(backbone, heads, lambda x: x, cfg, input_shape=(1, 3, 32, 32))


def test_wrapper_inference_on_cuda_returns_cuda_prediction():
    wrapper = _build(thresholds=(0.0, 0.0)).to("cuda")  # always exits at e0
    wrapper.eval()
    x = torch.randn(1, 3, 32, 32, device="cuda")
    result = wrapper(x, mode="inference")
    assert result.prediction.is_cuda
    assert result.exit_taken == 0
    assert result.computation_used < 1.0


def test_cuda_inference_is_grad_free_without_external_no_grad():
    """The inference_mode() fix must hold on CUDA too: no autograd graph on the
    returned prediction even without an external no_grad context."""
    wrapper = _build(thresholds=(1.0, 1.0)).to("cuda")
    wrapper.eval()
    x = torch.randn(1, 3, 32, 32, device="cuda")
    result = wrapper(x, mode="inference")
    assert result.prediction.requires_grad is False
    assert result.prediction.grad_fn is None


def test_benchmark_wrapper_cuda_sync_path_runs():
    """Exercises the ``device.startswith('cuda') -> torch.cuda.synchronize()``
    branch in the benchmark harness, never hit by the cpu suite."""
    wrapper = _build().to("cuda")
    r = benchmark_wrapper(
        wrapper, input_shape=(1, 3, 32, 32), device="cuda", num_warmup=3, num_runs=10
    )
    assert r.throughput_ips > 0
    assert r.latency_median_ms > 0
    assert r.device == "cuda"


def test_stage2_keeps_backbone_bn_frozen_on_cuda():
    """The documented BN-freeze invariant must hold on-device: stage 2 must not
    update the backbone's BatchNorm running statistics."""
    wrapper = _build().to("cuda")
    bn = next(m for m in wrapper.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d))
    before = bn.running_mean.clone()

    x = torch.randn(16, 3, 32, 32, device="cuda")
    y = torch.randint(0, 10, (16,), device="cuda")
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(TensorDataset(x.cpu(), y.cpu()), batch_size=8)
    stage2_train_exits(wrapper, loader, epochs=1, device="cuda", on_epoch_end=lambda _: None)

    assert torch.allclose(bn.running_mean, before), "stage 2 must not update backbone BN stats"
