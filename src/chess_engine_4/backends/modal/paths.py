"""Legacy Modal artifact locations, usable without importing the Modal SDK."""

from pathlib import Path

REMOTE_ARTIFACT_PATH = "/artifacts"
LC0_REMOTE_PATH = Path(REMOTE_ARTIFACT_PATH) / "bin" / "lc0-sm120"
LC0_SM80_REMOTE_PATH = Path(REMOTE_ARTIFACT_PATH) / "bin" / "lc0-sm80"
LC0_SM90_REMOTE_PATH = Path(REMOTE_ARTIFACT_PATH) / "bin" / "lc0-sm90"


def lc0_path_for_gpu(gpu: str) -> Path:
    if gpu == "A100":
        return LC0_SM80_REMOTE_PATH
    if gpu in {"H100", "H200"}:
        return LC0_SM90_REMOTE_PATH
    if gpu == "RTX-PRO-6000":
        return LC0_REMOTE_PATH
    raise ValueError(f"Unsupported evaluation GPU {gpu!r}.")
