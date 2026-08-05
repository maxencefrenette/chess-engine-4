# DeepGEMM MoE Throughput

## Goal

Estimate whether replacing the current Transformer Engine MoE expert operations
with DeepSeek's DeepGEMM grouped matrix multiplications would improve B200
training throughput.

The throwaway prototype preserved the canonical `moe64a2` architecture and
`B = 128d` batch scaling. It benchmarked a complete synthetic training step,
including packed-plane expansion, routing, alternating dense layers, output
heads, backward, and fused Adam. Data loading and host-to-device transfer were
excluded. Each result is the median of 50 measured steps after 10 warmup steps.

DeepGEMM was tested at commit `559d79fb6994a58b8a15b4b93bf13ccc16edf247`.

## Results

| Width | Batch | TE MXFP8 | DeepGEMM BF16 | DeepGEMM throughput |
| ---: | ---: | ---: | ---: | ---: |
| d128 | 16,384 | 19.63 ms | 7.48 ms | 2.62x |
| d256 | 32,768 | 21.55 ms | 15.29 ms | 1.41x |
| d512 | 65,536 | 41.50 ms | 40.62 ms | 1.02x |
| d1024 | 131,072 | 120.99 ms | 135.37 ms | 0.89x |

DeepGEMM substantially improved the small-width cases, approximately tied the
current implementation at `d512`, and was about 12% slower at `d1024`. This does
not support a wholesale backend migration because the canonical family is
intended to scale beyond the point where the prototype's advantage disappears.

## Limitations

The DeepGEMM prototype used BF16 grouped matrix multiplications. We did not test
DeepGEMM in FP8 mode. The comparison is therefore not precision-matched against
the current Transformer Engine MXFP8 implementation, and it does not establish
how an optimized DeepGEMM FP8 training path would perform.

DeepGEMM did not provide an autograd-ready MXFP8 training integration for this
prototype. A production implementation would also need efficient FP8 casting,
transposition, scaling, and backward integration rather than only replacing the
grouped GEMM calls.
