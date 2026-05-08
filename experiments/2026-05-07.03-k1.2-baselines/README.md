# k=1.2 Baselines

Date: 2026-05-07

## Goal

Reset the baseline runs after switching from physical-FLOPs budgets to step-adjusted compute budgets. The old `experiments/best-runs.toml` entries used the previous methodology, so these runs replace them even where the loss is not directly comparable.

The configs were shifted from `1e13`/`1e14`/`1e15` to `1e14`/`1e15`/`1e16`, with `step_penalty_k = 1.2`. The 10x compute-budget bump keeps step counts in roughly the same range under the new step penalty. The `1e16` baseline uses batch size 1024 to reduce wall-clock time.

## Commands

The launched commands are listed in `commands.txt`.

## Results

Metrics below use the current convention:

- `flops_seen` is actual measured training FLOPs.
- `compute_seen` is step-adjusted compute.
- `loss` and `policy_top1` are averages over the last 100 W&B history rows where both metrics are present.

| Budget | Shape | Params | Batch | LR | Steps | Samples | FLOPs seen | Compute seen | Final loss | Loss | Policy top-1 | Runtime | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1e14 | d48x1 | 463,094 | 192 | 0.001 | 23,528 | 4,517,376 | 1.34e13 | 1.00e14 | 4.6945 | 4.7959 | 0.2363 | 4.9m | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/dl7xy2lq) |
| 1e15 | d64x5 | 825,990 | 512 | 0.000707 | 44,227 | 22,644,224 | 1.18e14 | 1.00e15 | 4.3396 | 4.5407 | 0.2986 | 17.8m | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/14s9ljb8) |
| 1e16 | d192x4 | 3,506,246 | 1,024 | 0.0003 | 51,279 | 52,509,696 | 1.14e15 | 1.00e16 | 4.1945 | 4.2711 | 0.3530 | 37.4m | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/96u1lzqi) |

## Takeaways

- The new step-adjusted budgets landed where expected: each run ended at approximately its target `compute_budget`.
- The 10x budget bump with `k = 1.2` kept optimizer steps practical: about 24k, 44k, and 51k steps.
- These rows are now the scaling-law baselines in `experiments/best-runs.toml`.
