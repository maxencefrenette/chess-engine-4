# Dense 1e18 and 1e21 Retune

## Goal

Retune the two dense scaling points that were weak relative to neighboring
budgets. The sweep targeted model size, samples per parameter, batch size, and
learning rate while keeping each run at its existing compute budget.

Selection continues to minimize `loss_upper_1sd = loss + loss_std`.

## Interpolation

The `1e21` interpolation between the `1e20` and `1e22` winners suggested
roughly 23-27M parameters, 400-450M samples, and LR `4e-4` to `5e-4`. The old
run used 20.4M parameters, 513.9M samples, and LR `2e-4`.

At `1e18`, the old run used 35.2 samples per parameter, versus roughly 17-20
at the neighboring frontier. Fifteen valid runs tested widths 64-128, depths
2-4, batches 1,536-3,072, and LR `6e-4` to `1.4e-3`. Three additional `d80`
attempts failed before training because MXFP8 requires dimensions divisible by
32; they consumed no training compute.

## Results

| Budget | Recipe | Params | Samples | D/N | Loss | Loss + 1 SD | Policy top-1 | W&B |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `1e18` old | `d64x3`, batch 3,072, LR 1e-3 | 733,408 | 25,841,664 | 35.24 | 3.6192 | 3.7315 | 29.52% | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/vowzs4os) |
| `1e18` best mean | `d96x2`, batch 2,048, LR 1.2e-3 | 1,099,040 | 17,233,920 | 15.68 | **3.6009** | 3.7375 | **30.36%** | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/6kmyf9q2) |
| `1e18` selected | `d64x3`, batch 3,072, LR 1.4e-3 | 733,408 | 25,841,664 | 35.24 | 3.6100 | **3.7273** | 29.70% | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/vzu3es4b) |
| `1e21` old | `d512x5`, batch 32,768, LR 2e-4 | 20,403,616 | 513,900,544 | 25.19 | 3.0475 | 3.0761 | 44.69% | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/q46ovubj) |
| `1e21` selected | `d576x5`, batch 32,768, LR 4.5e-4 | 25,165,664 | 462,979,072 | 18.40 | **2.9950** | **3.0283** | **46.29%** | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/men5iops) |

All five `1e21` candidates improved mean loss over the old baseline:

| Model | Batch | LR | Samples | Loss | Loss + 1 SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `d576x5` | 24,576 | 4e-4 | 400,957,440 | 3.0053 | 3.0389 |
| `d576x5` | 24,576 | 5e-4 | 400,957,440 | 2.9985 | 3.0329 |
| `d640x4` | 24,576 | 4.5e-4 | 398,082,048 | 3.0073 | 3.0436 |
| `d576x5` | 32,768 | 4.5e-4 | 462,979,072 | **2.9950** | **3.0283** |
| `d640x4` | 32,768 | 4.5e-4 | 459,669,504 | 3.0015 | 3.0362 |

## Findings

- The `1e21` anomaly was primarily a learning-rate and allocation problem.
  Interpolation improved mean loss by 0.0526, upper-1-SD by 0.0478, and policy
  top-1 by 1.59 percentage points.
- The selected `1e21` run restores the expected allocation trend at 18.4
  samples per parameter.
- At `1e18`, the trend-aligned `d96x2` model has the best mean loss and policy
  accuracy, but its noisier tail prevents it from winning the established
  upper-1-SD criterion.
- The selected `1e18` change is therefore only an LR improvement. This point
  remains less structurally convincing than the larger-budget baselines.

The selected recipes now populate `configs/dense/`, and the dense scaling data
has been regenerated from `experiments/best-runs-dense.toml`.
