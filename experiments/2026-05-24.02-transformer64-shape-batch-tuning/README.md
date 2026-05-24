# Transformer64 Shape And Batch Tuning

The previous Transformer64 baselines saw less than one datapoint per parameter, so this sweep tested smaller models and larger batches under an $8 Modal budget.

The sweep ran in two stages: first `1e14` and `1e15` candidates in parallel, then the `1e16` candidates were selected around the smaller shapes that won the early budgets.

| Budget | Best Run | Model | Batch | Samples | Params | D/P | Loss | Policy Top-1 | W&B |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `1e14` | `t64shape-1e14-d48x1-b256` | `d48x1h4` | 256 | 861,696 | 52,948 | 16.27 | 4.0952 | 0.2400 | https://wandb.ai/maxence-frenette/chess-engine-4/runs/9x225o9e |
| `1e15` | `t64shape-1e15-d64x2-b512` | `d64x2h4` | 512 | 2,694,656 | 155,716 | 17.30 | 3.9463 | 0.2600 | https://wandb.ai/maxence-frenette/chess-engine-4/runs/50whbnt6 |
| `1e16` | `t64shape-1e16-d80x3-b1536` | `d80x3h4` | 1536 | 11,581,440 | 342,004 | 33.86 | 3.7577 | 0.2744 | https://wandb.ai/maxence-frenette/chess-engine-4/runs/81kcb2jv |

The new best runs replace the previous Transformer64 baselines in `experiments/best-runs-transformer64.toml`, and the `configs/transformer64/*.toml` files now point to these shapes.

The result is decisive: Transformer64 was substantially overparameterized at these compute budgets. The tuned `1e16` run has about 9% of the old parameters and improves tail loss from `5.1914` to `3.7577`.

Total W&B runtime across the 13 runs was about 1.9 GPU-hours on T4-class Modal workers, keeping the sweep below the requested budget cap.
