# JAX Transformer Engine MoE

## Goal

Test whether replacing the PyTorch MoE training stack with JAX and Transformer
Engine's JAX MoE block improves B200 training throughput for the canonical
`moe64a2` family. The throwaway implementation retained 64 experts, two active
experts, alternating dense and MoE layers, `B = 128d`, MXFP8 GEMMs, all three
chess heads, router loss, backward, global gradient clipping, and AdamW.

The benchmark used Transformer Engine 2.17 in NVIDIA's CUDA 13.3-compatible
`26.07` JAX container. The relevant JAX layer is the private, experimental
`_MoEBlock`. Its supported backward path required FP32 parameters and residuals
around the MXFP8 GEMMs; forcing the production model's BF16 parameter convention
caused fused RMSNorm backward dtype failures.

## Compute-Only Result

The first benchmark used resident synthetic inputs and measured 50 warmup plus
500 synchronized optimizer steps.

| Width | Parameters | PyTorch | JAX + TE MoE | Apparent speedup | PyTorch MFU | JAX MFU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d128 | 27.2M | 13.14 ms | 6.72 ms | 1.96x | 0.47% | 0.93% |
| d256 | 106.2M | 19.22 ms | 14.54 ms | 1.32x | 2.01% | 2.66% |
| d512 | 420.0M | 43.41 ms | 29.28 ms | 1.48x | 6.09% | 9.03% |
| d1024 | 1.671B | 133.64 ms | 133.09 ms | 1.00x | 14.50% | 14.56% |

This suggested that JAX reduced fixed dispatch and launch overhead at small and
medium widths, while offering no advantage once the d1024 expert GEMMs dominated
the step. It was not an end-to-end comparison: the synthetic input omitted the
production packed-plane expansion and real transfer boundary.

## Production Input Contract

The second benchmark added the actual Rust/Polars Parquet loader, eight prefetch
threads, packed planes, compact policy targets, GPU plane expansion, complete
policy/value/moves-left/router losses, and approximately 1,808 bytes of H2D
payload per sample.

| Width | PyTorch production | JAX resident real batch | JAX real H2D | End-to-end regression |
| ---: | ---: | ---: | ---: | ---: |
| d128 | 13.14 ms | 14.29 ms | 21.38 ms | 63% slower |
| d256 | 19.22 ms | 23.87 ms | 47.82 ms | 149% slower |
| d512 | 43.41 ms | 65.32 ms | 67.22 ms | 55% slower |
| d1024 | 133.64 ms | 146.56 ms | 146.21 ms | 9% slower |

The loader was not responsible: retrieving an already-prefetched Rust batch
took only `0.04-0.06 ms`. Manual pinned-memory staging was worse because it
introduced another CPU copy; d128 increased to `47.08 ms/step`.

Even the resident-real-batch path lost at every width. The synthetic benchmark's
apparent gain was therefore erased before H2D by work omitted from that harness,
especially packed-plane expansion, the complete losses, and JAX's required FP32
parameter/residual convention. Pageable H2D then added another large penalty at
small widths.

## Conclusion

Do not migrate MoE training to JAX. The real training contract is slower by 9%
at d1024 and substantially slower at smaller widths. The path would also add a
second framework, depend on a private Transformer Engine API, require a newer
CUDA container, and change parameter/residual precision semantics.

Revisit only if Transformer Engine publishes a stable JAX MoE API whose native
input and dtype path matches this project, or if it provides a way to fuse the
packed-input expansion and transfer boundary into the compiled step. No
throwaway benchmark code was retained.
