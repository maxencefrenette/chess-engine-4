# Dense More-Data Allocation

## Goal

Test whether the `1e21` and `1e22` dense baselines were allocating too much
compute to model parameters and too little to training samples. Selection still
minimizes `loss_upper_1sd = loss + loss_std`.

## Data

The first `1e22` pass exposed a hard dataset limit: all four runs stopped after
about 592M samples, before reaching their compute budgets. Those truncated runs
were excluded. The Modal volume was extended from 72 to 96 tar files by adding
all of `2024-04-11`, after which every selected comparison reached its full
budget.

## Results

| Budget | Recipe | Params | Samples | Loss + 1 SD | Policy top-1 | W&B |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `1e21` old | `d640x5`, batch 12,288, LR 2.5e-4 | 30,419,232 | 258,011,136 | 3.0897 | 44.92% | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/jbowyv42) |
| `1e21` new | `d512x5`, batch 32,768, LR 2e-4 | 20,403,616 | 513,900,544 | **3.0761** | 44.69% | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/q46ovubj) |
| `1e22` old | `d1280x6`, batch 9,216, LR 1.5e-4 | 129,650,592 | 343,277,568 | 2.9748 | 48.17% | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/llwm76ng) |
| `1e22` new | `d1152x6`, batch 24,576, LR 1.4e-4 | 106,068,896 | 619,585,536 | **2.9494** | **48.70%** | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/idabxu2m) |

## Commands

The selected runs are reproducible from the updated configs:

```bash
uv run train-modal --config configs/dense/1e21.toml
uv run train-modal --config configs/dense/1e22.toml
```

## Findings

- `1e21` benefits from a substantially smaller model and larger batch. Samples
  nearly doubled and the selection score improved by 0.0135.
- `1e22` shows the same effect more clearly. Samples increased by 80%, total
  parameters fell by 18%, and the selection score improved by 0.0253.
- Larger batch alone was not sufficient in the initial sweep. The useful change
  was reallocating compute jointly across model size, batch size, and learning
  rate.
- The `1e21` winner slightly reduced policy top-1 despite improving aggregate
  loss, while the `1e22` winner improved both metrics.
