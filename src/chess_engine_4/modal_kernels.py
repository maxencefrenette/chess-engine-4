"""Build and benchmark project-owned CUDA kernels on Modal."""

from __future__ import annotations

import argparse
import json
from typing import Any

from chess_engine_4.modal_train import app, base_image

KERNEL_NAME = "dense-d128-mxfp8-forward"
MIN_COSINE_SIMILARITY = 0.999
MAX_MEAN_ABSOLUTE_ERROR = 1e-3

kernel_image = (
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


def benchmark_kernel_modal() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a CUDA kernel on Modal.")
    parser.add_argument("--kernel", choices=(KERNEL_NAME,), default=KERNEL_NAME)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=2_000)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.batch_size % 256:
        parser.error("batch-size must be a positive multiple of 256")
    if args.warmup < 0:
        parser.error("warmup must be non-negative")
    if args.iterations <= 0:
        parser.error("iterations must be positive")

    with app.run():
        result = _benchmark_dense_d128_mxfp8.remote(
            args.batch_size,
            args.warmup,
            args.iterations,
        )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(
        f"kernel={KERNEL_NAME} batch_size={result['batch_size']} "
        f"custom_ms={result['custom_ms']:.4f} te_ms={result['te_ms']:.4f} "
        f"speedup={result['speedup_vs_te']:.3f}x"
    )
    print(
        f"mean_abs_error={result['mean_abs_error_vs_te']:.6f} "
        f"max_abs_error={result['max_abs_error_vs_te']:.6f} "
        f"cosine_similarity={result['cosine_similarity_vs_te']:.8f}"
    )


@app.function(image=kernel_image, gpu="B200", timeout=30 * 60)
def _benchmark_dense_d128_mxfp8(
    batch_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    import torch
    from torch.nn import functional as F

    from chess_engine_4.kernels import dense_d128_mxfp8_forward
    from chess_engine_4.model.dense import DenseBlock
    from chess_engine_4.model.transformer_engine import autocast_context

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    x = torch.randn(batch_size, 128, device="cuda", dtype=torch.bfloat16)
    block = DenseBlock(
        d_model=128,
        hidden_dim=512,
        rms_norm_eps=1e-6,
        activation="swiglu",
    ).cuda().eval()
    layer = block.layer
    norm_weight = layer.layer_norm_weight.detach()
    gate_up_weight = layer.fc1_weight.detach()
    down_weight = layer.fc2_weight.detach()

    with torch.no_grad(), autocast_context("mxfp8"):
        te_output = block(x)
    with torch.no_grad():
        custom_output = dense_d128_mxfp8_forward(
            x,
            norm_weight,
            gate_up_weight,
            down_weight,
        )
    torch.cuda.synchronize()

    difference = custom_output.float() - te_output.float()
    mean_abs_error = difference.abs().mean().item()
    max_abs_error = difference.abs().max().item()
    cosine_similarity = F.cosine_similarity(
        custom_output.float().flatten(),
        te_output.float().flatten(),
        dim=0,
    ).item()
    if not torch.isfinite(custom_output).all():
        raise RuntimeError("custom kernel produced non-finite output")
    if cosine_similarity < MIN_COSINE_SIMILARITY:
        raise RuntimeError(
            f"custom kernel cosine similarity {cosine_similarity:.8f} is below "
            f"{MIN_COSINE_SIMILARITY}"
        )
    if mean_abs_error > MAX_MEAN_ABSOLUTE_ERROR:
        raise RuntimeError(
            f"custom kernel mean absolute error {mean_abs_error:.8f} exceeds "
            f"{MAX_MEAN_ABSOLUTE_ERROR}"
        )

    def run_custom() -> None:
        dense_d128_mxfp8_forward(
            x,
            norm_weight,
            gate_up_weight,
            down_weight,
        )

    def run_te() -> None:
        with autocast_context("mxfp8"):
            block(x)

    custom_ms = _cuda_time(run_custom, warmup=warmup, iterations=iterations)
    te_ms = _cuda_time(run_te, warmup=warmup, iterations=iterations)
    return {
        "kernel": KERNEL_NAME,
        "batch_size": batch_size,
        "warmup": warmup,
        "iterations": iterations,
        "custom_ms": custom_ms,
        "te_ms": te_ms,
        "speedup_vs_te": te_ms / custom_ms,
        "mean_abs_error_vs_te": mean_abs_error,
        "max_abs_error_vs_te": max_abs_error,
        "cosine_similarity_vs_te": cosine_similarity,
        "device_name": torch.cuda.get_device_name(),
    }


def _cuda_time(function: Any, *, warmup: int, iterations: int) -> float:
    import torch

    with torch.no_grad():
        for _ in range(warmup):
            function()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        end.synchronize()
    return start.elapsed_time(end) / iterations
