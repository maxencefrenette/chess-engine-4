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

`dense-d128-mxfp8-forward` is the first development target. The current operator
uses ThunderKittens' MXFP8 quantization and GEMM kernels as a correct baseline;
the projection and activation launches are intentionally still separate so that
future fusion gains can be measured explicitly.
