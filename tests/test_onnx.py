"""ONNX export tests — the exported static graph must reproduce the wrapper's
all-exits (training-mode) outputs under onnxruntime."""

import numpy as np
import pytest
import torch

from earlyon.models import custom_ee
from earlyon.onnx import export_to_onnx
from tests.fixtures.tiny_models import TinyBackbone, TinyTokenBackbone

pytest.importorskip("onnx")  # export_to_onnx needs it; skip cleanly if absent
ort = pytest.importorskip("onnxruntime")


def _cnn_model():
    return custom_ee(
        TinyBackbone(num_classes=10),
        ["stage1", "stage2"],
        num_classes=10,
        input_shape=(1, 3, 32, 32),
    ).eval()


def _token_model():
    return custom_ee(
        TinyTokenBackbone(num_classes=10),
        ["block0", "block1"],
        num_classes=10,
        input_shape=(1, 3, 32, 32),
    ).eval()


def _run_onnx(path, x):
    session = ort.InferenceSession(str(path))
    return session.run(None, {"input": x.numpy()})


@pytest.mark.parametrize("builder", [_cnn_model, _token_model], ids=["cnn", "token"])
def test_export_outputs_match_torch(builder, tmp_path):
    """onnxruntime outputs equal the wrapper's training-mode logits — for both
    4D-conv and 3D-token exit heads."""
    model = builder()
    path = tmp_path / "m.onnx"
    names = export_to_onnx(model, path, input_shape=(1, 3, 32, 32))
    assert names == ["exit_0", "exit_1", "final"]
    assert path.exists()

    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        ref = model(x, mode="training")
    outs = _run_onnx(path, x)
    assert len(outs) == len(ref) == 3
    for o, r in zip(outs, ref):
        assert np.allclose(o, r.numpy(), atol=1e-4)


@pytest.mark.parametrize("builder", [_cnn_model, _token_model], ids=["cnn", "token"])
def test_export_dynamic_batch_accepts_other_batch_sizes(builder, tmp_path):
    """Traced at batch 1, the dynamic-batch graph must still run (and match) at
    batch 4 — for both conv and transformer traces."""
    model = builder()
    path = tmp_path / "dyn.onnx"
    export_to_onnx(model, path, input_shape=(1, 3, 32, 32), dynamic_batch=True)

    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        ref = model(x, mode="training")
    outs = _run_onnx(path, x)
    assert outs[0].shape == (4, 10)
    for o, r in zip(outs, ref):
        assert np.allclose(o, r.numpy(), atol=1e-4)


def test_export_static_batch_rejects_other_batch_sizes(tmp_path):
    """Without dynamic_batch the graph is fixed to the traced batch size."""
    model = _cnn_model()
    path = tmp_path / "static.onnx"
    export_to_onnx(model, path, input_shape=(1, 3, 32, 32), dynamic_batch=False)

    # batch 1 works
    outs = _run_onnx(path, torch.randn(1, 3, 32, 32))
    assert outs[0].shape == (1, 10)
    # batch 4 is rejected by the static graph (onnxruntime dimension mismatch)
    with pytest.raises(Exception, match=r"invalid dimensions|Got: 4|index"):
        _run_onnx(path, torch.randn(4, 3, 32, 32))


@pytest.mark.parametrize("start_training", [True, False])
def test_export_preserves_model_mode(start_training, tmp_path):
    """Export must not flip the model's train/eval mode (torch.onnx.export does;
    we restore it)."""
    model = _cnn_model()
    model.train(start_training)
    export_to_onnx(model, tmp_path / "m.onnx", input_shape=(1, 3, 32, 32))
    assert model.training is start_training


def _route_from_onnx(outs, thresholds, policy, temperature=1.0):
    """Reproduce the wrapper's routing rule from the exported all-exits outputs:
    return (chosen_exit_index, argmax_prediction)."""

    def softmax(z):
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z / temperature)
        return e / e.sum(axis=-1, keepdims=True)

    for i, logits in enumerate(outs[:-1]):
        p = softmax(logits)
        if policy == "confidence":
            if p.max() >= thresholds[i]:
                return i, int(logits.argmax())
        else:  # entropy
            ent = float(-(p * np.log(np.clip(p, 1e-12, None))).sum())
            if ent <= thresholds[i]:
                return i, int(logits.argmax())
    return -1, int(outs[-1].argmax())


@pytest.mark.parametrize("force_exit", [True, False], ids=["exit0", "final"])
def test_onnx_routing_matches_torch_confidence(force_exit, tmp_path):
    """The deployment contract: applying the confidence routing rule to the ONNX
    outputs reproduces the torch wrapper's inference decision and prediction."""
    model = _cnn_model()
    thr = [0.0, 0.0] if force_exit else [1.01, 1.01]
    model.config.confidence_thresholds = thr
    path = tmp_path / "m.onnx"
    export_to_onnx(model, path, input_shape=(1, 3, 32, 32))

    x = torch.randn(1, 3, 32, 32)
    idx, pred = _route_from_onnx(_run_onnx(path, x), thr, "confidence")
    with torch.no_grad():
        ref = model(x, mode="inference")
    assert idx == ref.exit_taken == (0 if force_exit else -1)
    assert pred == int(ref.prediction.argmax())


@pytest.mark.parametrize("force_exit", [True, False], ids=["exit0", "final"])
def test_onnx_routing_matches_torch_entropy(force_exit, tmp_path):
    """Same deployment contract for the entropy policy."""
    import math

    model = custom_ee(
        TinyBackbone(num_classes=10),
        ["stage1", "stage2"],
        num_classes=10,
        input_shape=(1, 3, 32, 32),
        routing_policy="entropy",
    )
    thr = [math.log(10) + 1.0] * 2 if force_exit else [0.0, 0.0]
    model.config.entropy_thresholds = thr
    model.eval()  # disable head dropout so the prediction is deterministic
    path = tmp_path / "e.onnx"
    export_to_onnx(model, path, input_shape=(1, 3, 32, 32))

    x = torch.randn(1, 3, 32, 32)
    idx, pred = _route_from_onnx(_run_onnx(path, x), thr, "entropy")
    with torch.no_grad():
        ref = model(x, mode="inference")
    assert idx == ref.exit_taken == (0 if force_exit else -1)
    assert pred == int(ref.prediction.argmax())


def test_export_output_count_tracks_exits(tmp_path):
    """A 3-exit model exports 4 outputs (exit_0..exit_2 + final)."""
    model = custom_ee(
        TinyBackbone(num_classes=10),
        ["stage1", "stage2", "stage3"],
        num_classes=10,
        input_shape=(1, 3, 32, 32),
    ).eval()
    names = export_to_onnx(model, tmp_path / "three.onnx", input_shape=(1, 3, 32, 32))
    assert names == ["exit_0", "exit_1", "exit_2", "final"]
