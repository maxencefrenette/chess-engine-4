"""Shared CUDA extension loading."""

from __future__ import annotations

import ctypes
import importlib
from functools import cache
from typing import Any


@cache
def extension() -> Any:
    try:
        ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
        return importlib.import_module("_chess_engine_4_kernels")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "The CUDA kernel extension is not built. Run `uv run build-kernels` on a "
            f"supported CUDA host. Import error: {exc}"
        ) from exc
