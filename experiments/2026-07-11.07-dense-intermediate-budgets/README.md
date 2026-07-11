# Dense Intermediate Compute Budgets

## Goal

Add and tune dense baselines at `3e18` and `3e19` compute so the scaling-law
fit is not determined only by decade-spaced observations. Each budget received
20 full Modal runs: eight architecture trials, ten joint batch/LR trials, and
two boundary checks selected from the preceding results.

Selection minimizes `loss_upper_1sd = loss + loss_std`.

## Initial Interpolation

Log-space interpolation between the neighboring best runs suggested:

| Budget | Parameters | Batch | LR | Initial practical model |
| --- | ---: | ---: | ---: | --- |
| `3e18` | 1.36M | 3,500 | 8.6e-4 | `d96x4`, batch 4,096, LR 8.5e-4 |
| `3e19` | 4.48M | 5,700 | 4.5e-4 | `d224x4`, batch 6,144, LR 4.5e-4 |

All tested widths were multiples of 32 as required by Transformer Engine
MXFP8.

## Selected Results

| Budget | Recipe | Params | Samples | D/N | Loss | Loss + 1 SD | Policy top-1 | W&B |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `3e18` initial | `d96x4`, batch 4,096, LR 8.5e-4 | 1,320,416 | 38,645,760 | 29.27 | 3.4954 | 3.5797 | 32.82% | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/tpt49xcv) |
| `3e18` selected | `d128x2`, batch 4,096, LR 1.45e-3 | 1,563,040 | 35,442,688 | 22.68 | **3.4743** | **3.5744** | **33.03%** | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/7ohug58s) |
| `3e19` initial | `d224x4`, batch 6,144, LR 4.5e-4 | 4,454,624 | 81,948,672 | 18.40 | 3.2943 | 3.3518 | 37.85% | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/1z7oovp5) |
| `3e19` selected | `d192x4`, batch 12,288, LR 6e-4 | 3,523,616 | 130,166,784 | 36.94 | **3.2662** | **3.3194** | **38.20%** | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/1gs863uo) |

## Sweep Findings

### 3e18

- `d128x2` was the best architecture under upper-1-SD. `d128x3` achieved a
  slightly better mean in the shape pass but had materially higher variance.
- Batch 4,096 beat both 3,072 and 6,144. The larger batch sharply increased the
  measured tail variance at this short scale.
- LR improved through `1.45e-3`, then turned over at `1.6e-3` and degraded
  clearly at `1.75e-3`.

### 3e19

- Architecture performance was nearly flat from `d192x4` through `d256x4` at
  the interpolated batch and LR. `d192x4` had the best upper-1-SD score.
- Increasing batch from 6,144 to 8,192 improved both mean and variance.
- The final batch-12,288 boundary check improved again and became the selected
  run. This winner lies on the batch boundary and has a high D/N ratio, so a
  future refinement should jointly retest larger models at this batch rather
  than assume the architecture optimum is fully settled.

The selected recipes are stored in `configs/dense/3e18.toml` and
`configs/dense/3e19.toml`, and both runs are included in
`experiments/best-runs-dense.toml`.
