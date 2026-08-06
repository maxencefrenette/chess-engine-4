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

The dense MXFP8 operator supports d128 through d2048. It shares TK's stock wide
GEMM across d256 and larger widths, specializes only the RMSNorm launch shape,
and retains a narrow GEMM specialization for d128's 128-column projections. The
forward and backward remain a short sequence of specialized launches; fusing
the two GEMMs into single persistent full-layer kernels is the next optimization
boundary. The wider forward path does not require per-width GEMM kernels. At
realistic training batches, however, its explicit backward remains slower than
Transformer Engine because quantization, weight-gradient GEMMs, activation, and
normalization remain separate launches; keep it behind
`--experimental-dense-kernel` until that path is fused further.

The backward path fuses transpose with MXFP8 quantization, avoiding six BF16
transpose materializations per block, and uses width-specialized BF16-pair
SwiGLU kernels. Run `uv run benchmark-kernel-modal --d-model WIDTH` to compare
forward and backward latency independently against Transformer Engine.
