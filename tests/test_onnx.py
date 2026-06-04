"""ONNX export tests — feature not yet supported (stub raises)."""

import pytest

from earlyon.onnx import export_to_onnx


def test_export_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not yet supported"):
        export_to_onnx(None, "/tmp/x")
