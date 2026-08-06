# Native Kernels

This directory contains Blackwell-only CUDA kernels used by the Python training
package and, eventually, the lc0 inference backend. ThunderKittens is pinned as a
git submodule under `third_party/ThunderKittens`.

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
