# MoE Per-Step EMA Validation

## Goal

Correct `loss/task[ema=0.99]` so it consumes every optimizer step rather than
only the loss observed at ten-step logging boundaries. Then rerun the old `32d`
incumbents and new canonical `128d` recipes at `d128`, `d256`, and `d512` with
matched data and training FLOPs.

The implementation reuses the per-step CPU loss values already transferred for
spike detection. It adds no synchronization, device transfer, CUDA kernel, or
W&B logging.

## Matched Results

| Width | Batch | Steps | LR | EMA loss | Final loss | Runtime | Spikes | Dead experts | W&B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| d128 | 4,096 | 6,630 | 1.2e-3 | **3.4141** | 3.4074 | 145s | 0 | 1 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/kddy9fj7) |
| d128 | 16,384 | 1,657 | 3.3e-3 | 3.4726 | 3.5220 | 28s | 0 | 0 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/s1nwf38y) |
| d256 | 8,192 | 12,966 | 4.3e-4 | **3.1578** | 3.1297 | 264s | 0 | 0 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/cowfao8n) |
| d256 | 32,768 | 3,241 | 1.2e-3 | 3.1824 | 3.1217 | 83s | 0 | 0 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/5kcj5a04) |
| d512 | 16,384 | 25,637 | 1.6e-4 | **2.9733** | 2.9764 | 607s | 0 | 0 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/ichs1d55) |
| d512 | 65,536 | 6,409 | 4.4e-4 | 2.9752 | 2.9778 | 330s | 0 | 0 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/64th71sp) |

At equal training FLOPs, `32d` remains better at `d128` and `d256`. The `d512`
results remain close: the EMA difference is `0.0019`, while `128d`
finishes 46% faster. Runtime improves by 81%, 69%, and 46% respectively as
width increases.

Against the corrected `32d` loss/FLOPs curve, the `128d` runs achieve
`EG_flops = 0.630x`, `0.753x`, and `0.968x`. This is the expected cost of using
four times fewer optimizer steps.

This experiment explicitly optimizes realized B200 training cost rather than
loss per FLOP. The `128d` runs are therefore **promoted** as the canonical MoE
baselines despite their lower `EG_flops`. They reduce measured runtime by 81%,
69%, and 46%, eliminate the dead expert at `d128`, and become close to
loss-neutral by `d512`. `experiments/best-runs-moe64a2.toml` now records these
cost-oriented defaults.

## Corrected LR Check

A focused `128d` sweep under the corrected EMA selected:

| Width | Best tested LR | EMA loss | Canonical LR | Canonical EMA loss | W&B |
| ---: | ---: | ---: | ---: | ---: | --- |
| d128 | 2.7e-3 | 3.4650 | 3.3e-3 | 3.4726 | [best](https://wandb.ai/maxence-frenette/chess-engine-4/runs/gb4djvis) |
| d256 | 1.2e-3 | 3.1824 | 1.2e-3 | 3.1824 | [best](https://wandb.ai/maxence-frenette/chess-engine-4/runs/97cpbhz0) |
| d512 | 4.5e-4 | 2.9714 | 4.4e-4 | 2.9750 | [best](https://wandb.ai/maxence-frenette/chess-engine-4/runs/b3q0hdkr) |

The existing coefficient `89` and exponent `-0.74` remain a reasonable smooth
law: its canonical values are exactly optimal at `d256` and within `0.008` loss
of the tested optimum at the other widths. Keeping the smooth law avoids three
bespoke width overrides.

## Policy EMA

Exact per-step policy top-1 was tested and rejected. A 500-step `d512/128d`
profile measured `43.35 ms/step` with it versus `42.72 ms/step` without it, a
1.45% wall-time regression for a diagnostic metric. Policy top-1 is therefore
computed only at the ten-step logging boundary and uses an EMA decay of `0.9`,
which gives it an approximately 100-step horizon. Loss EMA remains exact per
optimizer step.

## Commands

Representative matched commands:

```sh
uv run train-modal --config configs/moe64a2.py --d-model 256 \
  --batch-size 8192 --steps 12966 --lr 0.00043
uv run train-modal --config configs/moe64a2.py --d-model 256
```
