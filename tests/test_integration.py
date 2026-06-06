"""End-to-end integration tests — the pieces working *together*.

Each test drives a full pipeline (build → train → calibrate → analyze →
benchmark → save/load → ONNX) on a small model, across both routing policies and
both a factory backbone and a custom-wrapped one.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from earlyon.benchmarking import benchmark_wrapper_on_loader, evaluate
from earlyon.core.thresholds import calibrate_thresholds
from earlyon.models import cifar_resnet_ee, custom_ee
from earlyon.onnx import export_to_onnx
from earlyon.training import joint_train_backbone_and_exits, stage2_train_exits
from earlyon.utils import load_wrapper, save_wrapper
from tests.fixtures.tiny_models import TinyTokenBackbone

pytest.importorskip("onnx")  # export_to_onnx needs it; skip cleanly if absent
ort = pytest.importorskip("onnxruntime")


def _loader(n=24, bs=8):
    return DataLoader(
        TensorDataset(torch.randn(n, 3, 32, 32), torch.randint(0, 10, (n,))), batch_size=bs
    )


def _loader1(n=24):
    return DataLoader(
        TensorDataset(torch.randn(n, 3, 32, 32), torch.randint(0, 10, (n,))), batch_size=1
    )


def test_pipeline_cifar_resnet_confidence_train_calibrate_save_onnx(tmp_path):
    """Factory backbone, confidence routing: train exits → calibrate → analyze →
    benchmark → save/load (round-trips thresholds + routing) → ONNX (matches)."""
    model = cifar_resnet_ee(num_classes=10, depth=8)
    stage2_train_exits(model, _loader(), epochs=1, device="cpu", on_epoch_end=lambda _: None)

    result = calibrate_thresholds(model, _loader1(), target_accuracy_drop=0.5, device="cpu")
    assert result.policy == "confidence"
    assert model.config.confidence_thresholds == result.thresholds

    report = evaluate(model, _loader1(), device="cpu")
    assert 0.0 <= report.overall_accuracy <= 1.0
    assert abs(sum(report.exit_distribution.values()) - 1.0) < 1e-6

    bench = benchmark_wrapper_on_loader(model, _loader1(), device="cpu", num_warmup=2, num_runs=8)
    assert bench.throughput_ips > 0

    path = tmp_path / "m.pth"
    save_wrapper(model, path)
    loaded = load_wrapper(path)
    assert loaded.config.backbone == "cifar_resnet8"
    assert loaded.config.confidence_thresholds == result.thresholds

    # same weights + thresholds -> identical routing on a fixed input
    x = torch.randn(1, 3, 32, 32)
    model.eval()
    loaded.eval()
    with torch.no_grad():
        r1 = model(x, mode="inference")
        r2 = loaded(x, mode="inference")
    assert r1.exit_taken == r2.exit_taken
    assert torch.allclose(r1.prediction, r2.prediction, atol=1e-5)

    onnx_path = tmp_path / "m.onnx"
    export_to_onnx(loaded, onnx_path, input_shape=(1, 3, 32, 32))
    with torch.no_grad():
        ref = loaded(x, mode="training")
    outs = ort.InferenceSession(str(onnx_path)).run(None, {"input": x.numpy()})
    for o, r in zip(outs, ref):
        assert np.allclose(o, r.numpy(), atol=1e-4)


def test_pipeline_custom_token_entropy_joint_calibrate_onnx(tmp_path):
    """Custom-wrapped transformer, entropy routing: joint train → entropy
    calibrate (updates entropy_thresholds) → analyze → ONNX. Custom models are
    not load_wrapper-able, which the pipeline asserts."""
    model = custom_ee(
        TinyTokenBackbone(num_classes=10),
        ["block0", "block1"],
        num_classes=10,
        input_shape=(1, 3, 32, 32),
        routing_policy="entropy",
    )
    joint_train_backbone_and_exits(
        model, _loader(), epochs=1, device="cpu", on_epoch_end=lambda _: None
    )

    result = calibrate_thresholds(model, _loader1(), target_accuracy_drop=1.0, device="cpu")
    assert result.policy == "entropy"
    assert model.config.entropy_thresholds == result.thresholds

    report = evaluate(model, _loader1(), device="cpu")
    assert set(report.exit_distribution).issubset({"exit_0", "exit_1", "final"})

    # custom models save the state_dict but can't be rebuilt via load_wrapper
    save_wrapper(model, tmp_path / "c.pth")
    with pytest.raises(NotImplementedError, match="custom_ee"):
        load_wrapper(tmp_path / "c.pth")

    # ONNX export still works (the token/3D head path)
    onnx_path = tmp_path / "c.onnx"
    export_to_onnx(model, onnx_path, input_shape=(1, 3, 32, 32))
    x = torch.randn(1, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        ref = model(x, mode="training")
    outs = ort.InferenceSession(str(onnx_path)).run(None, {"input": x.numpy()})
    for o, r in zip(outs, ref):
        assert np.allclose(o, r.numpy(), atol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_pipeline_on_cuda_train_calibrate_analyze(tmp_path):
    """The whole train → calibrate → analyze pipeline runs on CUDA end to end."""
    model = cifar_resnet_ee(num_classes=10, depth=8)
    stage2_train_exits(model, _loader(), epochs=1, device="cuda", on_epoch_end=lambda _: None)
    result = calibrate_thresholds(model, _loader1(), target_accuracy_drop=0.5, device="cuda")
    assert len(result.thresholds) == len(model.config.exit_points)
    report = evaluate(model, _loader1(), device="cuda")
    assert 0.0 <= report.overall_accuracy <= 1.0
