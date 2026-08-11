# Batched Muon kernels and LR tuning

## Goal

Remove the serial small-matrix launch bottleneck in PyTorch 2.11 Muon, then
lightly retune learning rate at d256 and d512. This follows the initial Muon
pilot, which found training-FLOP gains through d256 but no realized cost gain.

## Implementation

The optimized path preserves PyTorch Muon's update rule, BF16 Newton-Schulz
iteration, five iteration steps, Nesterov momentum, and Adam-RMS learning-rate
adjustment. It changes execution only:

- equal oriented `fc1` and `fc2` matrices are stacked across eight blocks;
- Newton-Schulz uses batched GEMMs instead of processing 16 matrices serially;
- momentum, decay, and parameter updates use multi-tensor operations;
- `torch.compile` fuses the surrounding CUDA pointwise graph.

The numerical reference was the installed `torch.optim.Muon`. The performance
baseline was Transformer Engine FusedAdam at upstream commit
`07e281f2ba93b61c5ab6145dbdaa2a768b888e19`.

## Kernel benchmark

The temporary Muon implementation and benchmark code were removed after the
experiment. The benchmark performed one parity step and then timed 100 optimizer
steps after 10 warmup iterations.

| Width | GPU | Matrices | PyTorch Muon | Batched Muon | Speedup | Max BF16 parameter difference |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| d256 | RTX Pro 6000 | 16 | 2.883 ms | **0.583 ms** | **4.94x** | 0.000488 |
| d512 | B200 | 16 | 4.571 ms | **1.023 ms** | **4.47x** | 0.000488 |

Canonical synthetic TE-model steps with the batched optimizer measured 3.368 ms
at d256 and 6.396 ms at d512. More importantly, matched full training at the
same 0.5x LR fell from 43.02 to 34.97 seconds at d256 and from 114.27 to 51.93
seconds at d512. The optimizer-only benchmark therefore correctly predicted a
large end-to-end gain, although d256 retained more non-optimizer overhead than
the simple projection assumed.

## Learning-rate results

All runs used the same `0.055x` horizon, seed 1, accepted samples, batch, loss,
and schedule as the initial pilot. Every arm was spike-free.

| Width | Optimizer | LR multiplier | EMA loss | `EG_flops` | Runtime | Cost | W&B |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| d256 | AdamW | `1x` | 3.409629 | 0.926x | 23.84 s | $0.0226 | [ufjguuuy](https://wandb.ai/maxence-frenette/chess-engine-4/runs/ufjguuuy) |
| d256 | Batched Muon | `0.5x` | 3.354710 | 1.250x | 34.97 s | $0.0331 | [28dnm521](https://wandb.ai/maxence-frenette/chess-engine-4/runs/28dnm521) |
| d256 | Batched Muon | `0.75x` | 3.343279 | 1.334x | 35.04 s | $0.0332 | [p54fvb9z](https://wandb.ai/maxence-frenette/chess-engine-4/runs/p54fvb9z) |
| d256 | Batched Muon | `1x` | **3.337754** | **1.377x** | **32.28 s** | **$0.0306** | [nfo80tar](https://wandb.ai/maxence-frenette/uncategorized/runs/nfo80tar) |
| d512 | AdamW | `1x` | **3.168128** | **0.997x** | 46.73 s | $0.0860 | [u4c69pip](https://wandb.ai/maxence-frenette/chess-engine-4/runs/u4c69pip) |
| d512 | Batched Muon | `0.5x` | 3.191106 | 0.862x | 51.93 s | $0.0956 | [3qvzh4j0](https://wandb.ai/maxence-frenette/chess-engine-4/runs/3qvzh4j0) |
| d512 | Batched Muon | `0.75x` | 3.275905 | 0.521x | 45.43 s | $0.0836 | [9cdi11dn](https://wandb.ai/maxence-frenette/chess-engine-4/runs/9cdi11dn) |

At d256, `1x` is the selected tested Muon LR. Relative to AdamW, its
`EG_flops` is 48.7% higher while runtime is 35.4% higher, producing a **9.8%
realized dollar-efficiency gain**. This is the first tested scale where Muon
beats AdamW after runtime is included.

At d512, `0.5x` remains the selected tested Muon LR. Raising it to `0.75x`
degrades convergence sharply. Even the better arm has 13.5% lower `EG_flops`
and 11.1% higher runtime than AdamW, so realized efficiency remains 22.2%
lower. A higher LR has poor value of information.

The five new training arms cost $0.276 total. The d256 `1x` run was accidentally
logged to W&B's `uncategorized` project because it was launched directly without
the shared project environment; its configuration and metrics are otherwise
complete.

## Verdict

**The kernel optimization succeeds, and Muon realizes a small net gain at
d256.** Batching removes most of the implementation artifact that made the
initial pilot misleading on cost. Select `1x` at d256 and `0.5x` at d512 among
the tested arms, but do not promote Muon into the canonical dense recipe: the
d512 exception remains, and this short single-seed evidence is not enough for a
family-wide optimizer cutover.

A subsequent experiment should test d256 `1x` and d512 `0.5x` at a longer
horizon or additional seed before promotion. D512 architecture-specific tuning
may also be required, but another LR increase is not supported by this sweep.
