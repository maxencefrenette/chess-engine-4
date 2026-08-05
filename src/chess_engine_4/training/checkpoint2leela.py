"""Convert PyTorch checkpoints to lc0 ONNX weights files."""

from __future__ import annotations

import argparse
import gzip
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal
import onnx
import torch
from torch import nn
from torch.export import Dim

from chess_engine_4.modal_train import (
    ARTIFACT_VOLUME_NAME,
    REMOTE_ARTIFACT_PATH,
    artifact_volume,
    image,
)
from chess_engine_4.model import ChessNetOutput, build_model, model_config_from_dict
from chess_engine_4.model.transformer_engine import te

DEFAULT_ONNX_INPUT = "/input/planes"
DEFAULT_ONNX_OUTPUT_POLICY = "/output/policy"
DEFAULT_ONNX_OUTPUT_WDL = "/output/wdl"
DEFAULT_ONNX_OUTPUT_MLH = "/output/mlh"
DEFAULT_EXPORT_DTYPE = "fp32"
REMOTE_LEELA_PATH = Path(REMOTE_ARTIFACT_PATH) / "leela"

app = modal.App("chess-engine-4-export")


class LeelaOnnxWrapper(nn.Module):
    """Expose model outputs in the form expected by lc0's ONNX backend."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output: ChessNetOutput = self.model(planes)
        return (
            output.policy_logits,
            torch.softmax(output.wdl_logits, dim=-1),
            output.moves_left.unsqueeze(-1),
        )


def checkpoint2leela() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a chess-engine-4 checkpoint to an lc0 ONNX weights file."
    )
    parser.add_argument(
        "checkpoint",
        help="Checkpoint path in the chess-engine-4-artifacts Modal Volume.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--onnx-output", type=Path, default=None)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--export-dtype",
        choices=("fp16", "fp32"),
        default=DEFAULT_EXPORT_DTYPE,
    )
    parser.add_argument("--keep-onnx", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = (
        args.output or Path("artifacts/leela") / checkpoint_path.with_suffix(".pb.gz").name
    )
    onnx_path = args.onnx_output or _default_onnx_path(output_path)
    remote_output = REMOTE_LEELA_PATH / output_path.name
    remote_onnx = REMOTE_LEELA_PATH / onnx_path.name

    with app.run():
        result = _export_checkpoint_remote.remote(
            str(_mounted_artifact_path(checkpoint_path)),
            str(remote_onnx),
            str(remote_output),
            args.opset,
            args.export_dtype,
        )

    _download_artifact(Path(result["weights_path"]), output_path)
    if args.keep_onnx:
        _download_artifact(Path(result["onnx_path"]), onnx_path)
    if not args.keep_onnx:
        _remove_artifact(Path(result["onnx_path"]))
    print(f"wrote {output_path}")


def export_checkpoint_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    *,
    opset: int = 18,
    export_dtype: str = DEFAULT_EXPORT_DTYPE,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    model = _model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    dtype = _torch_export_dtype(export_dtype)
    wrapper = LeelaOnnxWrapper(model.to(dtype=dtype)).eval().cuda()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 112, 8, 8, dtype=dtype, device="cuda")

    transformer_engine = te()
    from transformer_engine.pytorch.export import te_translation_table

    with torch.no_grad(), transformer_engine.autocast(enabled=False):
        wrapper(example)
        with transformer_engine.onnx_export(enabled=True):
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
                dynamic_shapes=({0: Dim("batch")},),
                opset_version=opset,
                dynamo=True,
                custom_translation_table=te_translation_table,
                external_data=False,
            )


def convert_onnx_to_leela(
    *,
    onnx_path: Path,
    output_path: Path,
) -> None:
    data_type = validate_onnx_interface(onnx_path)
    onnx_bytes = onnx_path.read_bytes()
    leela_bytes = leela_onnx_net_bytes(
        onnx_bytes=onnx_bytes,
        data_type=data_type,
        input_name=DEFAULT_ONNX_INPUT,
        policy_output_name=DEFAULT_ONNX_OUTPUT_POLICY,
        wdl_output_name=DEFAULT_ONNX_OUTPUT_WDL,
        moves_left_output_name=DEFAULT_ONNX_OUTPUT_MLH,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wb") as handle:
        handle.write(leela_bytes)


def validate_onnx_interface(onnx_path: Path) -> int:
    model = onnx.load(onnx_path)
    custom_domains = sorted({node.domain for node in model.graph.node if node.domain})
    if custom_domains:
        raise ValueError(f"ONNX graph contains unexpected operator domains: {custom_domains}.")
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
    input_type = model.graph.input[0].type.tensor_type.elem_type
    output_types = {output.type.tensor_type.elem_type for output in model.graph.output}
    supported_types = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16}
    if input_type not in supported_types or output_types != {input_type}:
        raise ValueError(
            f"Expected matching FLOAT or FLOAT16 ONNX interface, "
            f"got input={input_type}, outputs={output_types}."
        )
    return input_type


def leela_onnx_net_bytes(
    *,
    onnx_bytes: bytes,
    data_type: int,
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
            _uint32_field(2, data_type),
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


def _model_from_checkpoint(checkpoint: dict[str, Any]) -> nn.Module:
    config = checkpoint.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Checkpoint does not contain a config.model mapping.")
    model_config = dict(config["model"])
    parsed_config = model_config_from_dict(model_config)
    if parsed_config.kind != "dense":
        raise ValueError("LC0 ONNX export currently supports only dense checkpoints.")
    if not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError("Checkpoint does not contain a model_state_dict mapping.")
    return build_model(parsed_config).cuda()


def _mounted_artifact_path(path: Path) -> Path:
    if str(path).startswith(f"{REMOTE_ARTIFACT_PATH}/"):
        return path
    return Path(REMOTE_ARTIFACT_PATH) / str(path).lstrip("/")


def _torch_export_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unknown export dtype: {name}")


def _volume_path(path: Path) -> str:
    return "/" + str(path).removeprefix(REMOTE_ARTIFACT_PATH).lstrip("/")


def _download_artifact(remote_path: Path, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "modal",
            "volume",
            "get",
            "--force",
            ARTIFACT_VOLUME_NAME,
            _volume_path(remote_path),
            str(local_path),
        ],
        check=True,
    )


def _remove_artifact(remote_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "modal",
            "volume",
            "rm",
            ARTIFACT_VOLUME_NAME,
            _volume_path(remote_path),
        ],
        check=True,
    )


@app.function(
    image=image,
    gpu="B200",
    volumes={REMOTE_ARTIFACT_PATH: artifact_volume},
    timeout=60 * 60,
)
def _export_checkpoint_remote(
    checkpoint_path: str,
    onnx_path: str,
    weights_path: str,
    opset: int,
    export_dtype: str,
) -> dict[str, str]:
    output_path = Path(onnx_path)
    weights_output_path = Path(weights_path)
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(checkpoint_path)
    export_checkpoint_to_onnx(
        Path(checkpoint_path),
        output_path,
        opset=opset,
        export_dtype=export_dtype,
    )
    convert_onnx_to_leela(onnx_path=output_path, output_path=weights_output_path)
    artifact_volume.commit()
    return {"onnx_path": str(output_path), "weights_path": str(weights_output_path)}


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
