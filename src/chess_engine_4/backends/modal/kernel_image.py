"""Modal image construction for project-owned CUDA kernels."""

from __future__ import annotations

import modal


def with_cuda_kernels(base_image: modal.Image) -> modal.Image:
    return (
        base_image.apt_install("cmake", "ninja-build")
        .uv_pip_install("pybind11>=3.0")
        .env(
            {
                "CUDA_HOME": "/.uv/.venv/lib/python3.14/site-packages/nvidia/cu13",
                "PATH": (
                    "/.uv/.venv/lib/python3.14/site-packages/nvidia/cu13/bin:"
                    "/.uv/.venv/bin:"
                    "/usr/local/bin:/usr/bin:/bin"
                ),
            }
        )
        .add_local_dir("kernels", remote_path="/root/kernels", copy=True)
        .add_local_dir("third_party", remote_path="/root/third_party", copy=True)
        .add_local_file(
            "src/chess_engine_4/kernels/build.py",
            remote_path="/root/build_kernels.py",
            copy=True,
        )
        .run_commands(
            "test -e $CUDA_HOME/lib64 || ln -s lib $CUDA_HOME/lib64",
            "test -e $CUDA_HOME/lib/libcudart.so || "
            "ln -s $CUDA_HOME/lib/libcudart.so.13 $CUDA_HOME/lib/libcudart.so",
            "cd /root && /.uv/.venv/bin/python /root/build_kernels.py "
            "--build-dir /root/kernels/build",
            "cp /root/kernels/build/_chess_engine_4_kernels*.so "
            "/.uv/.venv/lib/python3.14/site-packages/",
        )
        .add_local_python_source("chess_engine_4")
    )
