# SM120 dense training kernels

## Context

- Implementation commits: `0b36456d`, `d93016c8`
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120)
- Config: canonical dense d256 shape, BF16, batch 8192
- ThunderKittens: `1c3920d993404dd49a6d4c7267ea11d583bd5c68`
- Transformer Engine reference: `8260f49660cbadb78bc52c90449428c51625469d`
- W&B URL: N/A (focused benchmark; no training run)
- `EG_flops`: N/A (no loss curve or training run)

## Command

```sh
uv run benchmark-training-modal \
  --config configs/dense.py \
  --d-model 256 \
  --gpu RTX-PRO-6000 \
  --level all \
  --warmup 10 \
  --iterations 50 \
  --json
```

## Results

The initial SM120 warp-tiled GEMM implementation passed the numerical gates but
lost decisively to Transformer Engine. CUDA-graphed layer forward was 0.3532 ms
versus 0.0861 ms, layer backward was 0.3605 ms versus 0.1689 ms, and the real
pipeline production step was 8.6067 ms versus 5.3518 ms. Modal:
`ap-6ClDLQxlFvf6vnYmt4t7P3`.

Using the explicit SM120 binding with ATen's BF16 `mm_out` dispatch retained the
project-owned RMSNorm, SwiGLU, residual, and backward kernels while selecting
cuBLAS for GEMMs, consistent with the retained SM120 inference evidence. It
improved the final exact-code layer forward to 0.0992 ms and layer backward to
0.2806 ms, but still lost to Transformer Engine at 0.0870 ms and 0.1690 ms. The
matched synthetic training step was 4.6355 ms custom versus 3.6488 ms TE
(0.783x), and the real pipeline production step was 7.4324 ms custom versus
6.3739 ms TE (0.859x). Modal: `ap-VdDnP0Pj7ulBN1M7CslEwr`.

Forward output cosine similarity was 1.0000001 with mean absolute error
4.11e-08. The minimum gradient cosine similarity was 0.9999996; all gradients
were finite. The largest gradient mean absolute error was 5.12e-04 for the norm
weight, within the existing acceptance thresholds.

A reduced-atomic RMSNorm weight-gradient follow-up was rejected because it made
the matched layer backward slower (0.2930 ms custom versus 0.1731 ms TE) and did
not produce an end-to-end win. Modal: `ap-fzuzFLk5y7rJKXUYw2OpuH`.

## Verdict

Numerical correctness: pass.

Promotion: no. The SM120 BF16 implementation remains available only through
the explicit `custom` backend and canonical dense recipes remain on Transformer
Engine. The standalone SM120 dense inference path remains on cuBLAS and was not
modified.

Low-precision SM120 inference remains a separate follow-up because the current
export/runtime format does not represent the required quantized tensors.
