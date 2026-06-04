"""ONNX export for early-exit wrappers — not yet supported.

The honest story: ONNX has no clean answer for sample-dependent control
flow that satisfies all runtimes. We tried two approaches (per-exit static
graph shrapnel, and a TorchScript ``If`` op) and both ran into issues with
the new ``torch.export``-based exporter in torch 2.x.

A future approach is likely one of:
- ``torch.onnx.export`` with ``dynamo=False`` (legacy exporter) plus
  TorchScript ``If`` to express the routing as a single ONNX graph
- TensorRT backend bypass: skip ONNX entirely and emit a TRT engine

Until then, deploy the wrapper from PyTorch directly. This stub exists only
so existing imports do not break.
"""

from __future__ import annotations

from typing import NoReturn


def export_to_onnx(*args: object, **kwargs: object) -> NoReturn:
    raise NotImplementedError(
        "ONNX export is not yet supported: the torch 2.x onnx exporter rejects "
        "the dynamic control flow used by the early-exit wrapper. Deploy the "
        "wrapper directly from PyTorch; track the github issue for status."
    )
