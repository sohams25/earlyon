"""ONNX export tests — feature deferred to v0.2."""

import pytest

from earlyon.onnx import export_to_onnx


def test_export_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="v0.2"):
        export_to_onnx(None, "/tmp/x")
