"""Convert PyTorch checkpoints to lc0 ONNX weights files."""

from __future__ import annotations

import argparse
import gzip
import struct
from pathlib import Path
from typing import Any

import onnx
import torch
from torch import nn
from torch.export import Dim

from chess_engine_4.model import MlpChessNet, MlpChessNetConfig, MlpChessNetOutput

DEFAULT_ONNX_INPUT = "/input/planes"
DEFAULT_ONNX_OUTPUT_POLICY = "/output/policy"
DEFAULT_ONNX_OUTPUT_WDL = "/output/wdl"
DEFAULT_ONNX_OUTPUT_MLH = "/output/mlh"


class LeelaOnnxWrapper(nn.Module):
    """Expose model outputs in the form expected by lc0's ONNX backend."""

    def __init__(self, model: MlpChessNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output: MlpChessNetOutput = self.model(planes)
        return (
            output.policy_logits,
            torch.softmax(output.wdl_logits, dim=-1),
            output.moves_left.unsqueeze(-1),
        )


def checkpoint2leela() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a chess-engine-4 checkpoint to an lc0 ONNX weights file."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--onnx-output", type=Path, default=None)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--keep-onnx", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output_path = args.output or args.checkpoint.with_suffix(".pb.gz")
    onnx_path = args.onnx_output or _default_onnx_path(output_path)

    export_checkpoint_to_onnx(args.checkpoint, onnx_path, opset=args.opset)
    convert_onnx_to_leela(
        onnx_path=onnx_path,
        output_path=output_path,
    )
    if not args.keep_onnx:
        onnx_path.unlink()
    print(f"wrote {output_path}")


def export_checkpoint_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    *,
    opset: int = 18,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = _model_from_checkpoint(checkpoint)
    wrapper = LeelaOnnxWrapper(model).eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 112, 8, 8, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (example,),
        output_path,
        input_names=[DEFAULT_ONNX_INPUT],
        output_names=[
            DEFAULT_ONNX_OUTPUT_POLICY,
            DEFAULT_ONNX_OUTPUT_WDL,
            DEFAULT_ONNX_OUTPUT_MLH,
        ],
        dynamic_axes={
            DEFAULT_ONNX_INPUT: {0: "batch"},
            DEFAULT_ONNX_OUTPUT_POLICY: {0: "batch"},
            DEFAULT_ONNX_OUTPUT_WDL: {0: "batch"},
            DEFAULT_ONNX_OUTPUT_MLH: {0: "batch"},
        },
        dynamic_shapes=({0: Dim("batch")},),
        opset_version=opset,
        dynamo=True,
        external_data=False,
    )


def convert_onnx_to_leela(
    *,
    onnx_path: Path,
    output_path: Path,
) -> None:
    validate_onnx_interface(onnx_path)
    onnx_bytes = onnx_path.read_bytes()
    leela_bytes = leela_onnx_net_bytes(
        onnx_bytes=onnx_bytes,
        input_name=DEFAULT_ONNX_INPUT,
        policy_output_name=DEFAULT_ONNX_OUTPUT_POLICY,
        wdl_output_name=DEFAULT_ONNX_OUTPUT_WDL,
        moves_left_output_name=DEFAULT_ONNX_OUTPUT_MLH,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wb") as handle:
        handle.write(leela_bytes)


def validate_onnx_interface(onnx_path: Path) -> None:
    model = onnx.load(onnx_path)
    inputs = {value.name for value in model.graph.input}
    outputs = {value.name for value in model.graph.output}
    missing_inputs = {DEFAULT_ONNX_INPUT} - inputs
    missing_outputs = {
        DEFAULT_ONNX_OUTPUT_POLICY,
        DEFAULT_ONNX_OUTPUT_WDL,
        DEFAULT_ONNX_OUTPUT_MLH,
    } - outputs
    if missing_inputs or missing_outputs:
        raise ValueError(
            f"ONNX interface mismatch. Missing inputs={sorted(missing_inputs)}, "
            f"missing outputs={sorted(missing_outputs)}."
        )
    onnx.checker.check_model(model)


def leela_onnx_net_bytes(
    *,
    onnx_bytes: bytes,
    input_name: str,
    policy_output_name: str,
    wdl_output_name: str,
    moves_left_output_name: str,
) -> bytes:
    """Build the lc0 pblczero.Net protobuf wrapper for an ONNX model.

    This writes the small subset of lc0's proto2 schema needed by ONNX nets:
    Net, EngineVersion, Format, NetworkFormat, and OnnxModel.
    """

    network_format = b"".join(
        [
            _uint32_field(1, 1),  # INPUT_CLASSICAL_112_PLANE
            _uint32_field(3, 5),  # NETWORK_ONNX
            _uint32_field(4, 1),  # POLICY_CLASSICAL
            _uint32_field(5, 2),  # VALUE_WDL
            _uint32_field(6, 1),  # MOVES_LEFT_V1
        ]
    )
    format_message = _message_field(2, network_format)
    onnx_model = b"".join(
        [
            _bytes_field(1, onnx_bytes),
            _uint32_field(2, 1),  # FLOAT
            _string_field(3, input_name),
            _string_field(5, wdl_output_name),
            _string_field(6, policy_output_name),
            _string_field(7, moves_left_output_name),
        ]
    )
    min_version = b"".join(
        [
            _uint32_field(1, 0),
            _uint32_field(2, 28),
        ]
    )
    return b"".join(
        [
            _fixed32_field(1, 0x1C0),
            _message_field(3, min_version),
            _message_field(4, format_message),
            _message_field(11, onnx_model),
        ]
    )


def _model_from_checkpoint(checkpoint: dict[str, Any]) -> MlpChessNet:
    config = checkpoint.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Checkpoint does not contain a config.model mapping.")
    model = MlpChessNet(MlpChessNetConfig(**config["model"]))
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a model_state_dict mapping.")
    model.load_state_dict(state_dict)
    return model


def _default_onnx_path(output_path: Path) -> Path:
    if output_path.name.endswith(".pb.gz"):
        return output_path.with_name(output_path.name.removesuffix(".pb.gz") + ".onnx")
    return output_path.with_suffix(".onnx")


def _uint32_field(field_number: int, value: int) -> bytes:
    return _key(field_number, 0) + _varint(value)


def _fixed32_field(field_number: int, value: int) -> bytes:
    return _key(field_number, 5) + struct.pack("<I", value)


def _string_field(field_number: int, value: str) -> bytes:
    return _bytes_field(field_number, value.encode("utf-8"))


def _bytes_field(field_number: int, value: bytes) -> bytes:
    return _key(field_number, 2) + _varint(len(value)) + value


def _message_field(field_number: int, value: bytes) -> bytes:
    return _bytes_field(field_number, value)


def _key(field_number: int, wire_type: int) -> bytes:
    return _varint((field_number << 3) | wire_type)


def _varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint value must be non-negative.")
    encoded = bytearray()
    while value > 0x7F:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


if __name__ == "__main__":
    checkpoint2leela()
