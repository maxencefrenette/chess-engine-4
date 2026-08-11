# Muon optimizer pilot

## Goal

Test whether Muon improves training-FLOP efficiency over fused AdamW for the
stacked dense MLP, and whether any convergence gain survives optimizer runtime
overhead as model width increases.

Muon was applied only to the 2-D weight matrices in `blocks.*`. The input
projection, output heads, norms, and biases stayed on Transformer Engine fused
AdamW. PyTorch 2.11's Muon implementation used its default momentum, Nesterov,
five Newton-Schulz steps, and `adjust_lr_fn="match_rms_adamw"`. The comparison
kept the model, seed, data, batch, steps, loss, schedule, and training FLOPs
matched to the completed dense `0.055x` width ladder.

The runs were based on commit `8cc8fd49bddfe7538f1fe50f3baedafb53b99d7f`
with the Muon implementation and this experiment present in the worktree.

The temporary Muon implementation and launch code were removed after the
experiment. The results and W&B run links are retained below.

## Cheap LR screen

The d64 screen tested `0.5x`, `1x`, and `2x` the matched AdamW learning rate.
All three runs were spike-free. The `0.5x` arm had the lowest EMA loss and the
highest `EG_flops`, although it was effectively tied with `1x`.

| Optimizer | LR multiplier | EMA loss | `EG_flops` | Runtime | Cost | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| AdamW | `1x` | 3.889359 | 1.128x | 10.06 s | $0.0095 | [adt7rfr5](https://wandb.ai/maxence-frenette/chess-engine-4/runs/adt7rfr5) |
| Muon | `0.5x` | **3.860349** | **1.271x** | 19.04 s | $0.0180 | [urfet6gy](https://wandb.ai/maxence-frenette/chess-engine-4/runs/urfet6gy) |
| Muon | `1x` | 3.861108 | 1.267x | 20.45 s | $0.0194 | [tbh0z3z6](https://wandb.ai/maxence-frenette/chess-engine-4/runs/tbh0z3z6) |
| Muon | `2x` | 3.870762 | 1.217x | 24.90 s | $0.0236 | [j07dm07p](https://wandb.ai/maxence-frenette/chess-engine-4/runs/j07dm07p) |

## Matched width results

The selected `0.5x` Muon LR multiplier was carried unchanged across widths.
`EG_flops` values below were recomputed for both optimizers against the same
current canonical fit, rather than copied from the earlier width report.

| Width | Adam loss / `EG_flops` | Muon loss / `EG_flops` | Relative EG gain | Runtime ratio | Realized efficiency change | Muon W&B |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| d64 | 3.889359 / 1.128x | **3.860349 / 1.271x** | +12.7% | 1.89x | -40.4% | [urfet6gy](https://wandb.ai/maxence-frenette/chess-engine-4/runs/urfet6gy) |
| d128 | 3.668101 / 0.917x | **3.614167 / 1.174x** | +28.0% | 1.96x | -34.6% | [tym0ilj5](https://wandb.ai/maxence-frenette/chess-engine-4/runs/tym0ilj5) |
| d256 | 3.409629 / 0.926x | **3.353811 / 1.256x** | **+35.6%** | 1.80x | **-24.8%** | [ibxacy4k](https://wandb.ai/maxence-frenette/chess-engine-4/runs/ibxacy4k) |
| d512 | **3.168128 / 0.997x** | 3.173265 / 0.965x | -3.2% | 2.45x | -60.4% | [8yc97of8](https://wandb.ai/maxence-frenette/chess-engine-4/runs/8yc97of8) |

Realized efficiency is the relative `EG_flops` gain divided by the relative
runtime at fixed hardware price. Muon improved convergence at three of four
widths, with the largest gain at d256, but the stock PyTorch implementation was
too slow to improve realized dollar efficiency at any tested width. Its d256
runtime was 43.02 seconds versus 23.84 seconds for AdamW. D512 regressed both
convergence and runtime at the selected transferred LR.

W&B system telemetry did not show a meaningful memory penalty in the pairs for
which both runs retained samples: d256 reported 5.76 GB Muon versus 5.75 GB
AdamW, and d512 reported 2.72 GB versus 2.89 GB. These are coarse sampled
allocations, not allocator peak measurements, so the safe conclusion is only
that memory was not the limiting issue in this pilot.

The six Muon arms cost $0.338 in total. All were spike-free.

## Verdict

**Muon shows a real training-FLOP-efficiency signal, but is not ready to replace
fused AdamW.** The hypothesis was supported through d256: Muon improved
`EG_flops` at d64, d128, and d256, and the gain grew to 35.6% at d256. The
realized-efficiency hypothesis was not supported: optimizer overhead remained
larger than the convergence gain, even at d256, and d512 slightly regressed.

Do not promote Muon into the canonical recipe. The next useful test is an
optimized/batched Muon kernel or compiled optimizer step, benchmarked first at
d256 and d512; further training runs with the current scalar PyTorch optimizer
path have poor value of information.

## Methodology note

PyTorch documents `match_rms_adamw` as the Muon scaling intended to reuse an
AdamW-tuned learning rate. That motivated the narrow LR screen. The measured
losses, `EG_flops`, overhead, memory telemetry, and verdict are project evidence,
not claims from the cited Muon work.
