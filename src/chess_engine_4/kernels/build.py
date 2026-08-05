"""Build the project-owned CUDA extension."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def build_kernels() -> None:
    parser = argparse.ArgumentParser(description="Build the Blackwell CUDA kernels.")
    parser.add_argument("--build-dir", type=Path, default=Path("kernels/build"))
    args = parser.parse_args()

    try:
        import pybind11
        import torch
    except ImportError as exc:
        raise RuntimeError("building kernels requires torch and pybind11") from exc

    cwd = Path.cwd()
    if (cwd / "kernels" / "CMakeLists.txt").exists():
        root = cwd
    else:
        root = Path(__file__).resolve().parents[3]
    source_dir = root / "kernels"
    build_dir = (root / args.build_dir).resolve()
    prefix_path = ";".join((torch.utils.cmake_prefix_path, pybind11.get_cmake_dir()))
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    build_env = os.environ.copy()
    build_env["TORCH_CUDA_ARCH_LIST"] = "10.0a"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            "-G",
            "Ninja",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DCMAKE_PREFIX_PATH={prefix_path}",
            f"-DCUDAToolkit_ROOT={cuda_home}",
            f"-DCUDA_TOOLKIT_ROOT_DIR={cuda_home}",
            f"-DCUDA_CUDART_LIBRARY={cuda_home / 'lib' / 'libcudart.so.13'}",
            f"-DCUDA_nvrtc_LIBRARY={cuda_home / 'lib' / 'libnvrtc.so.13'}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
        env=build_env,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel"],
        check=True,
        env=build_env,
    )
    print(build_dir)


if __name__ == "__main__":
    build_kernels()
