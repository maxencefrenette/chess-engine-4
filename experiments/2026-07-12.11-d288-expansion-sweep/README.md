# d288 Expansion Sweep

## Goal

Repeat the d128 expansion-factor sweep at `d288` to test whether wider residual
streams shift the preferred SwiGLU expansion. Candidates change only
`model.expansion_ratio`; the canonical 4x run is reused as control.

Selection uses `loss_upper_1sd` versus modified compute, with physical-FLOPs
efficiency as secondary context.

## Results

| Expansion | Params | Physical FLOPs | Loss | Loss + 1 SD | Policy top-1 | Modified efficiency | Physical efficiency | W&B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1x | 3,874,688 | 4.534e15 | 3.2258 | 3.2680 | 39.14% | 0.915x | 0.886x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/9pdg2qc5) |
| 2x | 5,118,848 | 5.935e15 | 3.1814 | 3.2258 | 40.41% | 1.090x | 1.088x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/1o93y99y) |
| **4x** | **7,607,168** | **8.737e15** | **3.1441** | **3.1873** | **41.85%** | **1.131x** | **1.116x** | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/oisz5ib0) |
| 8x | 12,583,808 | 1.434e16 | 3.1161 | 3.1659 | 42.45% | 0.879x | 0.936x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/3out01w1) |

As at d128, absolute loss and policy accuracy improve with expansion, but 8x is
not compute-efficient. The 2x run beats the global modified-compute trend yet
still trails the same-width 4x control. The 1x model is clearly too narrow.

Both tested widths therefore support retaining 4x expansion as the canonical
dense architecture.

## Commands

```bash
uv run train-modal --config configs/dense/d288.toml --expansion-ratio 1 --wandb-name d288-expansion-1x
uv run train-modal --config configs/dense/d288.toml --expansion-ratio 2 --wandb-name d288-expansion-2x
uv run train-modal --config configs/dense/d288.toml --expansion-ratio 8 --wandb-name d288-expansion-8x
```

The three candidate runs ran concurrently. `configs/dense/d288.toml` remains at
4x expansion.
