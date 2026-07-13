# d128 Expansion Sweep

## Goal

Test whether the canonical 4x SwiGLU expansion is over- or under-sized at
`d128`. The candidates change only `model.expansion_ratio`; width, depth, batch
size, steps, learning rate, data, precision, and loss recipe remain fixed.

Selection uses `loss_upper_1sd` versus modified compute because changing the
expansion reallocates compute between model capacity and the fixed run recipe.
Physical-FLOPs efficiency is included as secondary context.

## Results

| Expansion | Params | Physical FLOPs | Loss | Loss + 1 SD | Policy top-1 | Modified efficiency | Physical efficiency | W&B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1x | 1,366,688 | 7.067e14 | 3.4214 | 3.4843 | 33.97% | 0.896x | 0.846x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/ixjgrzsc) |
| 2x | 1,563,296 | 8.036e14 | 3.3931 | 3.4536 | 34.81% | 1.024x | 0.964x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/l3ke7mrz) |
| **4x** | **1,956,512** | **9.976e14** | **3.3629** | **3.4260** | **35.78%** | **1.050x** | **1.030x** | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/903a3xgb) |
| 8x | 2,742,944 | 1.386e15 | 3.3396 | 3.4014 | 36.44% | 0.942x | 0.926x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/ho2q2hyo) |

The wider MLPs improve absolute loss and policy accuracy, but 8x does not repay
its additional compute. The 2x model is efficient enough to beat the global
modified-compute trend, but it remains behind the canonical 4x model at the same
width. The 1x model is clearly too narrow.

## Commands

```bash
uv run train-modal --config configs/dense/d128.toml --expansion-ratio 1 --wandb-name d128-expansion-1x
uv run train-modal --config configs/dense/d128.toml --expansion-ratio 2 --wandb-name d128-expansion-2x
uv run train-modal --config configs/dense/d128.toml --expansion-ratio 8 --wandb-name d128-expansion-8x
```

The three candidate runs ran concurrently. The existing canonical 4x run was
reused as the control. `configs/dense/d128.toml` remains at 4x expansion.
