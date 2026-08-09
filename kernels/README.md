# Native Kernels

This directory contains architecture-specific CUDA kernels used by the Python
training package and lc0 inference backend. ThunderKittens is pinned as a git
submodule under `third_party/ThunderKittens`.

Build the PyTorch adapter on a CUDA host with:

```sh
git submodule update --init
uv run build-kernels
```

The CUDA implementation is kept separate from model selection. New kernels must
first pass their reference checks and beat the retained implementation in a Modal
benchmark before the canonical model dispatches to them.

The custom dense operator supports d32 through d2048. d32 through d512 use TK's
Blackwell BF16 GEMM because MXFP8 quantization overhead dominates at those
shapes. d1024 and d2048 use TK's MXFP8 GEMM. RMSNorm and SwiGLU launches are
specialized for every supported width.

The `sm80` and `sm90` implementations provide portable TK warp-tiled BF16 dense
and MoE kernels for A100 and Hopper training. Standalone SM90 lc0 inference uses
cuBLAS for ordinary dense GEMMs and the project-owned BF16 MoE dispatch/expert
path. The Python extension dispatches by CUDA capability, while standalone lc0
builds select architecture 80, 90a, 100a, or 120a at build time.

The forward and backward remain a short sequence of specialized launches;
fusing the two GEMMs into single persistent full-layer kernels is the next
optimization boundary. At realistic training batches, the explicit backward
can remain slower than Transformer Engine because quantization, weight-gradient
GEMMs, activation, and normalization are separate launches. Keep the path behind
`--kernel-backend custom` until each width wins end-to-end.

The kernel has two precision regimes: BF16 through d512 and MXFP8 from d1024.
All widths use width-specialized BF16-pair SwiGLU kernels. Run
`uv run benchmark-training-modal --d-model WIDTH --level layer` to compare
CUDA-graphed forward and backward latency independently against Transformer
Engine.
