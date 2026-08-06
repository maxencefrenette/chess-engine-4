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

The custom dense operator supports d32 through d2048. MXFP8 tensor-core tiles
are 128 elements wide, so d32 and d64 use TK's Blackwell BF16 GEMM with native
32- and 64-element tiles. d128 retains its narrow MXFP8 GEMM specialization;
d256 and larger share TK's stock wide MXFP8 GEMM. RMSNorm and SwiGLU launches
are specialized for every supported width.

The forward and backward remain a short sequence of specialized launches;
fusing the two GEMMs into single persistent full-layer kernels is the next
optimization boundary. At realistic training batches, the explicit backward
can remain slower than Transformer Engine because quantization, weight-gradient
GEMMs, activation, and normalization are separate launches. Keep the path behind
`--experimental-dense-kernel` until each width wins end-to-end.

The kernel has three precision regimes. d32 and d64 use BF16 forward and
backward GEMMs, d128 through d512 use MXFP8 forward and BF16 backward GEMMs, and
d1024 and d2048 use MXFP8 throughout. All widths use width-specialized BF16-pair
SwiGLU kernels. Run
`uv run benchmark-training-modal --d-model WIDTH --level layer` to compare
CUDA-graphed forward and backward latency independently against Transformer
Engine.
