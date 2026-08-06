"""Shared CUDA extension loading."""

from __future__ import annotations

import importlib
from functools import cache
from typing import Any


@cache
def extension() -> Any:
    try:
        return importlib.import_module("_chess_engine_4_kernels")
    except ImportError as exc:
        raise RuntimeError(
            "The CUDA kernel extension is not built. Run `uv run build-kernels` on a "
            f"Blackwell CUDA host. Import error: {exc}"
        ) from exc
