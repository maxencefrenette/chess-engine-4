# Dense 1e19 and 1e20 Retune

## Goal

Retune the decade points that became inconsistent after adding the `3e18` and
`3e19` baselines. Each budget received ten runs: four architecture trials,
five joint batch/LR trials, and one adaptive LR boundary check.

Selection minimizes `loss_upper_1sd = loss + loss_std`.

## Interpolation

The updated neighboring points suggested:

| Budget | Parameters | Batch | LR |
| --- | ---: | ---: | ---: |
| `1e19` | about 2.4M | 7-8K | about 9e-4 |
| `1e20` | about 7.1M | 16-18K | about 5.4e-4 |

## Selected Results

| Budget | Recipe | Params | Samples | D/N | Loss | Loss + 1 SD | Policy top-1 | W&B |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `1e19` old | `d160x4`, batch 4,096, LR 5e-4 | 2,690,912 | 49,586,176 | 18.43 | 3.3894 | 3.4640 | 35.15% | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/aunrxaba) |
| `1e19` selected | `d128x4`, batch 8,192, LR 1.1e-3 | 1,956,512 | 82,116,608 | 41.97 | **3.3629** | **3.4260** | **35.78%** | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/903a3xgb) |
| `1e20` old | `d320x4`, batch 8,192, LR 4e-4 | 7,837,472 | 130,564,096 | 16.66 | 3.1950 | 3.2430 | 40.74% | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/rd27qx3g) |
| `1e20` selected | `d288x5`, batch 16,384, LR 7.5e-4 | 7,607,168 | 187,564,032 | 24.66 | **3.1441** | **3.1873** | **41.85%** | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/oisz5ib0) |

## Findings

### 1e19

- `d128x4` had the best upper-1-SD architecture result. Larger models reduced
  mean loss slightly but increased variance enough to lose the selection
  metric.
- Batch 8,192 beat 6,144, reallocating substantially more compute to data.
- LR improved through `1.1e-3` and turned over at `1.3e-3`.
- The selected run improves loss by 0.0265 and upper-1-SD by 0.0380.

### 1e20

- `d288x5` narrowly beat `d320x4`; both confirm the interpolated model-size
  regime.
- Batch 16,384 beat 12,288.
- LR continued improving through the final `7.5e-4` boundary check. This is a
  selected boundary value, so a later refinement could test a slightly higher
  LR if this point becomes important again.
- The selected run improves loss by 0.0508, upper-1-SD by 0.0557, and policy
  top-1 by 1.11 percentage points.

The selected recipes now populate `configs/dense/1e19.toml` and
`configs/dense/1e20.toml`, and both entries have been updated in
`experiments/best-runs-dense.toml`.
