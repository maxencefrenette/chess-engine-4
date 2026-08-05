# MoE 64A2 Baseline

## Goal

Establish the first scaling baseline for the alternating-layer `moe64a2` family
from `d32` through `d512`. All runs use the family recipe at `0.02x` Chinchilla
so subsequent MoE experiments can be run cheaply.

The five runs executed concurrently on Modal B200s. Their W&B runtimes sum to
1,267 seconds, or approximately `$2.20` at `$6.25` per B200-hour.

## Results

| Width | Parameters | Training FLOPs | Loss | Policy top-1 | Runtime | Dead experts | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| d32 | 1.92M | 4.82e12 | 4.7350 | 16.84% | 20s | 21 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/tehjrg0s) |
| d64 | 7.09M | 4.38e13 | 3.9898 | 24.69% | 40s | 7 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/z6aiew5u) |
| d128 | 27.16M | 4.64e14 | 3.4808 | 33.79% | 89s | 1 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/g6cmtfl6) |
| d256 | 106.21M | 5.63e15 | 3.1961 | 41.17% | 239s | 0 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/tcpfrhs0) |
| d512 | 420.04M | 7.63e16 | 2.9911 | 47.23% | 879s | 0 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/43oqkrvp) |

All runs completed with zero detected loss spikes. The small models did not use
every expert at the final logged step: dead-expert count falls from 21 at `d32`
to zero at `d256`. These runs remain the canonical width baselines requested for
the family, but router utilization is a constraint to revisit when tuning the
small end of the scaling ladder.

The canonical inputs are stored in
`experiments/best-runs-moe64a2.toml`. Final checkpoints were retained for every
run in the Modal artifact volume.

## Command

```sh
uv run train-modal --config configs/moe64a2.py --d-model 256 --wandb
```
