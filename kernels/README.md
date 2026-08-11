# Native Kernels

This directory contains architecture-specific CUDA kernels used by the Python
training package and lc0 inference backend. ThunderKittens is vendored as a git
subtree under `third_party/ThunderKittens`, pinned to upstream commit
`1c3920d993404dd49a6d4c7267ea11d583bd5c68`.

Build the PyTorch adapter on a CUDA host with:

```sh
uv run build-kernels
```

Update the vendored source with:

```sh
git subtree pull --prefix third_party/ThunderKittens \
  https://github.com/HazyResearch/ThunderKittens.git main --squash
```

The CUDA implementation is kept separate from model selection. New kernels must
first pass their reference checks and beat the retained implementation in a Modal
benchmark before the canonical model dispatches to them.

The custom dense operator supports the canonical d64 through d1280 ladder. d64 through d512 use TK's
Blackwell BF16 GEMM because MXFP8 quantization overhead dominates at those
shapes. d768 through d1280 use TK's MXFP8 GEMM. RMSNorm and SwiGLU launches are
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

The kernel has two precision regimes: BF16 through d512 and MXFP8 from d768.
All widths use width-specialized BF16-pair SwiGLU kernels. Run
`uv run benchmark-training-modal --d-model WIDTH --level layer` to compare
CUDA-graphed forward and backward latency independently against Transformer
Engine.
