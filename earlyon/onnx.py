"""ONNX export for early-exit wrappers.

ONNX has no clean, portable way to express sample-dependent control flow, so
earlyon does not try to bake the *routing* into the graph. Instead it exports a
single **static multi-output** graph that computes every exit:

    inputs:  input            (N, C, H, W)
    outputs: exit_0 .. exit_{k-1}, final   each (N, num_classes)

The deploying application runs the graph and applies the same routing rule
earlyon uses at runtime — take the first exit whose softmax max ≥ threshold
(or whose entropy ≤ threshold) — picking the prediction and stopping. The graph
always computes all exits, so this trades the compute saving for a fully
portable, runtime-agnostic artifact; use the PyTorch wrapper directly when you
need the actual early-exit speedup.
"""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from earlyon.core.wrappers import EarlyExitWrapper


class _AllExitsModule(nn.Module):
    """Adapter exposing the wrapper's all-exits training forward as a plain
    tuple-returning module, so the legacy ONNX tracer can capture a static graph.
    """

    def __init__(self, wrapper: EarlyExitWrapper) -> None:
        super().__init__()
        self.wrapper = wrapper

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = self.wrapper(x, mode="training")
        return tuple(outputs)


def export_to_onnx(
    model: EarlyExitWrapper,
    path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    opset: int = 17,
    dynamic_batch: bool = True,
) -> list[str]:
    """Export ``model`` to a static multi-output ONNX graph and return the output
    names (``["exit_0", ..., "exit_{k-1}", "final"]``).

    The graph computes every exit (routing is applied by the caller at runtime —
    see the module docstring). Uses the legacy TorchScript exporter
    (``dynamo=False``): the all-exits forward is static, so it traces cleanly,
    whereas the routing forward's dynamic control flow does not.

    Parameters
    ----------
    model:
        A trained :class:`EarlyExitWrapper`.
    path:
        Destination ``.onnx`` file.
    input_shape:
        Example input shape for tracing (and the static shape unless
        ``dynamic_batch``).
    opset:
        ONNX opset version.
    dynamic_batch:
        If True, mark the batch dimension of the input and every output as
        dynamic so the graph accepts any batch size.

    Raises ``ImportError`` if the ``onnx`` package is missing (install with
    ``pip install 'earlyon[onnx]'``) — the legacy exporter loads it at call time.
    """
    try:
        import onnx  # noqa: F401  # the legacy ONNX exporter imports it internally
    except ImportError as exc:
        raise ImportError(
            "ONNX export requires the 'onnx' package: pip install 'earlyon[onnx]'"
        ) from exc

    n_exits = len(model.config.exit_points)
    output_names = [f"exit_{i}" for i in range(n_exits)] + ["final"]

    dynamic_axes: dict[str, dict[int, str]] | None = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}}
        dynamic_axes.update({name: {0: "batch"} for name in output_names})

    # torch.onnx.export leaves the module in train mode afterwards; save and
    # restore the caller's mode so export has no surprising side effect (a
    # train-mode model would re-enable dropout / update BatchNorm on next call).
    was_training = model.training
    model.eval()
    adapter = _AllExitsModule(model)  # after eval() so the adapter is eval too
    dummy = torch.zeros(input_shape)

    export_kwargs: dict[str, Any] = {
        "input_names": ["input"],
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "opset_version": opset,
    }
    # ``dynamo`` only exists on torch>=2.4; older torch has the legacy
    # TorchScript exporter as the only option, so omitting it is equivalent.
    # On torch>=2.9 the new exporter is the default and trips the wrapper's
    # _is_compiling() guard, so we must force the legacy path with dynamo=False.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    try:
        with warnings.catch_warnings():
            # the legacy exporter is deprecated on torch>=2.9 but is the only
            # path that traces the wrapper; silence its expected noise.
            warnings.simplefilter("ignore", DeprecationWarning)
            torch.onnx.export(adapter, (dummy,), str(path), **export_kwargs)
    finally:
        model.train(was_training)
    return output_names
