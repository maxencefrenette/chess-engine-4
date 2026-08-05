# DeepGEMM MoE Training

## Goal

Test whether replacing the current Transformer Engine expert implementation
with DeepSeek's DeepGEMM kernels improves B200 training throughput for the
canonical `moe64a2` family. The throwaway benchmark held routing, token
permutation, 64 experts, two active experts, alternating dense and MoE layers,
`B = 128d`, heads, backward, and optimizer work constant.

DeepGEMM exposes grouped forward, activation-gradient, and weight-gradient
kernels on SM100, but it is not an autograd-ready MoE layer. The prototype had
to invoke forward, dgrad, and k-grouped wgrad explicitly. DeepGEMM also leaves
FP8 casting and transposition to the caller, and its supplied PyTorch casting
helpers are not optimized.

## Environment

The repository's benchmark-only CUDA toolchain initially paired CUDA 13.3
NVCC/CCCL with CUDA 13.0 runtime headers, which DeepGEMM's JIT rejected. The
throwaway image used matching CUDA 13.0 compiler and CCCL packages. Grouped
forward, dgrad, and wgrad passed numerical checks on B200 before timing.

JIT compilation was excluded from measurements. Small and medium widths used
graph-captured expert work to match the production CUDA-graph path; d1024 used
eager dynamic dispatch, matching the canonical implementation at that width.

## Results

The final synthetic full-step benchmark included routing, alternating dense
layers, all three heads, backward, and fused Adam. It excluded dataloading and
H2D transfer.

| Width | Batch | Current TE MXFP8 | DeepGEMM BF16 | Throughput change |
| ---: | ---: | ---: | ---: | ---: |
| d128 | 16,384 | 19.63 ms | 7.48 ms | 2.62x faster |
| d256 | 32,768 | 21.55 ms | 15.29 ms | 1.41x faster |
| d512 | 65,536 | 41.50 ms | 40.62 ms | 1.02x faster |
| d1024 | 131,072 | 120.99 ms | 135.37 ms | 12% slower |

The crossover remained after progressively improving the harness from eager
expert-only timings to graph-captured expert work and finally the synthetic
full model step. DeepGEMM substantially reduced fixed overhead for small expert
matrices, tied Transformer Engine around d512, and regressed once the d1024
expert GEMMs dominated.

## Limitations

The executable DeepGEMM prototype used BF16 grouped kernels, while production
uses Transformer Engine MXFP8. This is not a precision-matched replacement.
The experiment did not try DeepGEMM's FP8 kernels, so it does not establish how
an FP8 DeepGEMM implementation would compare with the current MXFP8 path.
Building a fair MXFP8 training path would require custom optimized activation
and gradient quantization, transposition, autograd integration, and likely
additional fusion. A kernel-only MXFP8 number would overstate achievable
training throughput by omitting those costs.

The benchmark therefore answers whether DeepGEMM's currently usable training
primitives justify a migration, not whether a future custom DeepGEMM-based MoE
stack could ever outperform Transformer Engine.

## Conclusion

Do not migrate to DeepGEMM. Its advantage disappears by d512 and reverses at
d1024, the scale most relevant to future MoE runs. The available integration
would also replace the current high-level MXFP8 path with custom BF16 autograd
and kernel plumbing.

Revisit if DeepGEMM ships an integrated MXFP8 autograd layer, optimized casting
and transposition, or materially better large-matrix SM100 kernels. A
width-specific backend for small MoEs is not worthwhile while those widths are
used primarily as inexpensive scaling probes. No throwaway benchmark code was
retained.
