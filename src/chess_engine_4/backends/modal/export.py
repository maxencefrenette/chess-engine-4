"""Explicit remote checkpoint export through the Modal artifacts volume."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

from chess_engine_4.backends.modal.train import (
    ARTIFACT_VOLUME_NAME,
    REMOTE_ARTIFACT_PATH,
    artifact_volume,
    image,
)
from chess_engine_4.release.export import export_checkpoint

REMOTE_MODEL_PATH = Path(REMOTE_ARTIFACT_PATH) / "models"
app = modal.App("chess-engine-4-export-model")


def export_remote_checkpoint(checkpoint: Path, output: Path, *, remote_only: bool) -> Path:
    remote_checkpoint = _mounted_artifact_path(checkpoint)
    remote_output = REMOTE_MODEL_PATH / output.name
    with app.run():
        result = Path(_export_checkpoint_remote.remote(str(remote_checkpoint), str(remote_output)))
    if remote_only:
        return result
    _download_artifact(result, output)
    return output


def _mounted_artifact_path(path: Path) -> Path:
    if str(path).startswith(f"{REMOTE_ARTIFACT_PATH}/"):
        return path
    return Path(REMOTE_ARTIFACT_PATH) / str(path).lstrip("/")


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


@app.function(
    image=image,
    cpu=2,
    volumes={REMOTE_ARTIFACT_PATH: artifact_volume},
    timeout=60 * 60,
)
def _export_checkpoint_remote(checkpoint_path: str, output_path: str) -> str:
    checkpoint = Path(checkpoint_path)
    output = Path(output_path)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    export_checkpoint(checkpoint, output)
    artifact_volume.commit()
    return str(output)
