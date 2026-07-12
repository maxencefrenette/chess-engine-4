# MoE Training and Deployment Feasibility

This throwaway experiment tested whether a top-2 MLP MoE can provide substantially
more model capacity at approximately fixed active arithmetic, how training speed
scales with the number of experts, and whether the resulting model can be deployed
through lc0. No MoE implementation or model artifacts were retained in the live
codebase.

## Setup

All training profiles used one B200 with MXFP8, batch size 131,072, `d1472x8`,
50 warmup steps, and 200 measured steps. The dense control used a 4x SwiGLU
expansion. Each MoE expert used a 2x expansion with two active experts, matching
the dense block's active expert arithmetic.

The MoE path used Transformer Engine's fused permutation, 256-row MXFP8 padding,
`GroupedLinear -> ScaledSwiGLU -> GroupedLinear` operation-fuser path, and fused
unpadding and unpermutation. The experimental single-parameter expert-weight
layout was disabled because it crashed during fused Adam; this did not disable
the fused expert kernels.

## Expert-count benchmark

| Model | Total parameters | Parameters vs dense | Training FLOPs/sample | GPU train time/step | Slowdown vs dense | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 221.5M | 1.00x | 1.3336B | **139.2 ms** | 1.00x | ~942K samples/s |
| 4E/2A | 429.8M | 1.94x | ~1.334B | **281.2 ms** | 2.02x | 454K samples/s |
| 8E/2A | 845.9M | 3.82x | 1.3343B | **288.1 ms** | 2.07x | 447K samples/s |
| 16E/2A | 1.678B | 7.58x | ~1.334B | **296.9 ms** | 2.13x | 435K samples/s |
| 32E/2A | 3.342B | 15.09x | ~1.334B | **320.2 ms** | 2.30x | 403K samples/s |

Data loading exposed less than 0.3 ms of GPU idle time per step, so the MoE
slowdown was not caused by the loader. At matched counted active FLOPs, the 8E/2A
model reached 13.5% train-only MFU versus 27.9% for dense.

The large fixed cost is entering sparse grouped execution at all. Once that cost
is paid, additional experts are relatively cheap:

| Change | Parameter increase | GPU-time increase |
| --- | ---: | ---: |
| 4E to 8E | 1.97x | 2.5% |
| 8E to 16E | 1.98x | 3.0% |
| 16E to 32E | 1.99x | 7.8% |
| 4E to 32E | 7.8x | 13.9% |

The remaining penalty comes from routing, permutation and unpermutation, less
efficient grouped GEMMs, per-expert MXFP8 padding, eager dispatch, and Adam work
over all expert parameters. The FLOP counter intentionally measures active
forward/backward arithmetic and therefore does not capture all of those costs.

## CUDA graph follow-up

An 8E/2A variant used fixed-capacity permutation buffers, device-resident expert
counts, aligned routing-map padding, and CUDA-graphed forward and backward without
dropping tokens.

| 8E/2A implementation | End-to-end time/step | GPU train time/step | Change vs dynamic eager |
| --- | ---: | ---: | ---: |
| Dynamic eager | 292.9 ms | 288.1 ms | baseline |
| Static buffers, eager | 292.6 ms | 285.5 ms | 0.1% faster end-to-end |
| Static buffers, CUDA graph | **287.1 ms** | **281.7 ms** | **2.0% faster end-to-end** |

The improvement is too small to justify retaining the static-buffer machinery.
Paged activation stashing also does not address this single-GPU workload: the
global routed-token count is always `batch_size * active_experts`, and only its
distribution among experts changes. The distributed receive-buffer fragmentation
that paged stashing targets is absent here.

## Deployment smoke test

A small `d128x2`, 16E/2A model was trained for 100 steps on 819,200 positions.
It exported as a portable FP32 ONNX graph containing standard top-2 routing,
dynamic expert-weight gathering, and selected-expert matrix multiplications.

Lc0 loaded two independent copies through `onnx-trt` on an L4, TensorRT generated
a real 17.6 MB `sm89` engine, and two policy-mode games completed without reported
CPU fallback. This establishes functional compatibility, not efficient sparse
inference at production scale.

TensorRT 11 has a native
[`IMoELayer`](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/transformers-moe.html)
for Blackwell-class GPUs, and NVIDIA reports substantial native MoE optimization
in its [TensorRT 11 release notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.0.0.html).
There is no evidence that the generic ONNX `TopK -> Gather -> MatMul` graph is
automatically converted to that layer. A large-model inference benchmark remains
necessary before treating the portable ONNX path as performant.

## Conclusion

MoE capacity is not free: even the smallest tested MoE took about twice the dense
GPU time at matched active arithmetic. However, 16E/2A provides 7.6x the total
parameters for only 2.13x the dense step time and just 3% more time than 8E/2A.
It is therefore the most attractive next quality experiment: materially more
capacity than 8E without the greater data-per-expert and routing risk of 32E.

The next decision should be based on training quality or Elo, not another kernel
microbenchmark. If 16E/2A cannot beat spending roughly twice as much compute on
dense training, the additional implementation and deployment complexity is not
justified. If it does, optimized inference remains solvable through native
TensorRT MoE on Blackwell or a purpose-built lc0 CUDA backend with portable,
versioned model weights.
