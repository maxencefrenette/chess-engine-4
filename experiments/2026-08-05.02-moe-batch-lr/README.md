# MoE 128d Batch Learning Rates

> Superseded by `experiments/2026-08-05.03-moe-step-ema`. This initial sweep
> updated the loss EMA only at logging boundaries, which made its values depend
> on the number of optimizer steps. The raw runs and learning-rate search remain
> useful historical context, but the reported EMA comparisons are not current.

## Goal

Promote `B = 128d` to the canonical `moe64a2` recipe and retune learning rates
at `d128`, `d256`, and `d512`. Runs held model shape, `0.02x` Chinchilla data,
loss weights, seed, and optimizer settings fixed.

The sweep started with `{1x, 1.5x, 2x, 3x, 4x}` of the previous `32d` learning
rate, then added narrow follow-up points around the lowest stable loss. A run was
eligible only with zero detected loss spikes and zero dead experts at the final
logged step.

## Selected Recipe

| Width | Batch | Steps | Learning rate | EMA loss | Final batch loss | Policy top-1 | Runtime | W&B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| d128 | 16,384 | 1,657 | 3.3e-3 | 4.4390 | 3.5220 | 25.18% | 32s | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/06j0q7qc) |
| d256 | 32,768 | 3,241 | 1.2e-3 | 3.4764 | 3.1217 | 37.28% | 88s | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/kngmxrag) |
| d512 | 65,536 | 6,409 | 4.4e-4 | 3.0345 | 2.9713 | 45.70% | 303s | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/xlpgu7zv) |

The `d512` table row uses the exact canonical `4.4e-4` run. The nearby
`4.5e-4` candidate was marginally better at `3.0312` EMA loss and remained
stable, but the difference was too small to justify breaking the smooth recipe.

The retained learning-rate law keeps the previous parameter exponent and raises
its coefficient from `31.75` to `89`. At `0.02x` Chinchilla, it resolves to the
three learning rates above after two-significant-digit rounding.

## Stability Boundary

- `d128`: lower EMA losses appeared near `3.0e-3` to `3.6e-3`, but some points
  left one or two experts dead. `3.3e-3` was the best nearby zero-dead-expert run.
- `d256`: `1.72e-3` had a nearly identical EMA loss to `1.2e-3`; intermediate
  and higher candidates started producing loss spikes. The smooth-law value
  had slightly better policy top-1 and remained fully stable.
- `d512`: EMA loss continued improving above `4.5e-4`, but every candidate from
  `4.8e-4` through `6.4e-4` had a detected loss spike.

## Promotion

The recipe change is intentional for realized throughput, but these runs do not
replace `experiments/best-runs-moe64a2.toml`. Against the existing
training-FLOPs curve, their `EG_flops` values are `0.008x`, `0.085x`, and
`0.528x`, respectively, so the formal promotion verdict is **keep
incumbent** at all three widths.

The comparison is especially harsh at small widths because
`loss/task[ema=0.99]` updates only on ten-step logging boundaries. Cutting the
number of optimizer steps by four leaves substantially more early-training loss
in the final EMA even when the final batch loss is close to the old run. This
does not affect comparisons within a width in this sweep because every LR
candidate used the same step count, but it prevents a clean EMA comparison with
the old `32d` runs.

## Commands

Representative selected commands:

```sh
uv run train-modal --config configs/moe64a2.py --d-model 128 --lr 0.0033
uv run train-modal --config configs/moe64a2.py --d-model 256 --lr 0.0012
uv run train-modal --config configs/moe64a2.py --d-model 512 --lr 0.00044
```
