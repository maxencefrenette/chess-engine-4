from pathlib import Path

import pytest
import torch

from chess_engine_4.training.checkpoint2leela import (
    DEFAULT_EXPORT_DTYPE,
    _default_onnx_path,
    _mounted_artifact_path,
    _torch_export_dtype,
    _volume_path,
)


def test_modal_artifact_paths() -> None:
    assert _mounted_artifact_path(Path("checkpoints/run-final.pt")) == Path(
        "/artifacts/checkpoints/run-final.pt"
    )
    assert _mounted_artifact_path(Path("/checkpoints/run-final.pt")) == Path(
        "/artifacts/checkpoints/run-final.pt"
    )
    assert _mounted_artifact_path(Path("/artifacts/checkpoints/run-final.pt")) == Path(
        "/artifacts/checkpoints/run-final.pt"
    )
    assert _volume_path(Path("/artifacts/leela/run.onnx")) == "/leela/run.onnx"
    assert _default_onnx_path(Path("artifacts/leela/run.pb.gz")) == Path("artifacts/leela/run.onnx")


def test_export_dtype() -> None:
    assert DEFAULT_EXPORT_DTYPE == "fp32"
    assert _torch_export_dtype("fp16") == torch.float16
    assert _torch_export_dtype("fp32") == torch.float32
    with pytest.raises(ValueError, match="unknown export dtype"):
        _torch_export_dtype("bf16")
