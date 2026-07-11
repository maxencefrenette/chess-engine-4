# Dense Blackwell Scaling Baselines

## Goal

Reset the dense scaling-law baselines after the B200 and Transformer Engine
MXFP8 cutover. The search lightly tunes model shape, batch size, and learning
rate at compute budgets from `1e18` through `1e22`.

Selection minimizes `loss_upper_1sd = loss + loss_std`, where the mean and
second moment are the final `0.99` EMA summaries logged to W&B.

## Scope

- 40 Modal B200 jobs were launched: 5 seeds, 20 shape candidates, and 15
  batch/LR candidates.
- 39 jobs completed with selection metrics. `d1024x4` at `1e21` timed out
  waiting for the dataloader while the Modal data volume was being extended.
- All training used the default Transformer Engine MXFP8 recipe.
- The Modal data volume was extended from 48 to 72 tar files by adding all of
  `2024-04-10`.

Every invocation is recorded in [commands.txt](commands.txt), and all W&B
results are in [results.csv](results.csv).

## Selected Baselines

| Budget | Model | Params | Batch | LR | Samples | Loss | Loss + 1 SD | Policy top-1 | W&B |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `1e18` | `d64x3` | 733,408 | 3,072 | 1e-3 | 25,841,664 | 3.6192 | 3.7315 | 0.2952 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/vowzs4os) |
| `1e19` | `d160x4` | 2,690,912 | 4,096 | 5e-4 | 49,586,176 | 3.3894 | 3.4640 | 0.3515 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/aunrxaba) |
| `1e20` | `d320x4` | 7,837,472 | 8,192 | 4e-4 | 130,564,096 | 3.1950 | 3.2430 | 0.4074 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/rd27qx3g) |
| `1e21` | `d640x5` | 30,419,232 | 12,288 | 2.5e-4 | 258,011,136 | 3.0475 | 3.0897 | 0.4492 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/jbowyv42) |
| `1e22` | `d1280x6` | 129,650,592 | 9,216 | 1.5e-4 | 343,277,568 | 2.9401 | 2.9748 | 0.4817 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/llwm76ng) |

## Fitted Trends

The five selected points produce the following provisional fits:

```text
L(C) = 2.5251 + 88.92 * C^-0.1061
params = 7.257e-05 * C^0.5548
D_samples = 128.3 * C^0.2963
batch_size = 8.927 * C^0.1431
lr = 2.97 * C^-0.1949
```

The loss-fit RMSE on the selected points is `0.0037`. Extrapolation to `1e23`
currently suggests approximately `d2048x8`, batch 16,384, and LR `1e-4`.

## Findings

- The higher-budget winners are smaller than the previous dense baselines:
  `d160` instead of `d192`, `d320` instead of `d384`, and `d640` instead of
  `d768`.
- `1e22` strongly preferred `d1280x6` over the larger `d1472`-`d1792`
  candidates. This is the clearest shape result in the sweep.
- Smaller batches remained competitive despite the step penalty. At `1e22`,
  batch 9,216 with LR `1.5e-4` beat batch 12,288 with LR `1e-4`.
- These are tuned starting points, not converged optima. Several winners sit
  near a tested boundary, especially the `1e22` learning rate.

The selected runs now populate `configs/dense/` and
`experiments/best-runs-dense.toml`.
