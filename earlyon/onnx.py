"""ONNX export for early-exit wrappers — DEFERRED to v0.2.

The honest story: ONNX has no clean answer for sample-dependent control
flow that satisfies all runtimes. We tried two approaches in v0.1
(per-exit static graph shrapnel, and TorchScript ``If`` op) and both ran
into issues with the new ``torch.export``-based exporter in torch 2.x.

The v0.2 plan is one of:
- ``torch.onnx.export`` with ``dynamo=False`` (legacy exporter) plus
  TorchScript ``If`` to express the routing as a single ONNX graph
- TensorRT backend bypass: skip ONNX entirely and emit a TRT engine

Until then, deploy the wrapper from PyTorch directly. The architect-led
v0.1 review explicitly cut ONNX from scope; this stub is here only so
existing imports do not break.
"""

from __future__ import annotations


def export_to_onnx(*args, **kwargs):
    raise NotImplementedError(
        "ONNX export was deferred to v0.2. The torch 2.x onnx exporter "
        "rejects the dynamic control flow used by the early-exit wrapper. "
        "Track the v0.2 issue on github for status."
    )
