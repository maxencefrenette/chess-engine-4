# Transformer64 Baselines

This establishes the first Transformer64 baseline points at `1e14`, `1e15`, and `1e16` compute budgets.

The runs use vanilla attention over 64 board-square tokens, learned square embeddings, pooled value and moves-left heads, and the LC0-style attention policy head. Metrics use the standard post-hoc W&B tail methodology over the last 100 logged values where available.

| Budget | Model | Batch | LR | Samples | Loss | Policy Top-1 | Runtime | W&B |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `1e14` | `d96x2h4` | 192 | 0.001 | 178,560 | 5.8743 | 0.2185 | 25.9s | https://wandb.ai/maxence-frenette/chess-engine-4/runs/e7s9p9rb |
| `1e15` | `d128x4h4` | 512 | 0.000707 | 538,624 | 5.5442 | 0.2314 | 135.8s | https://wandb.ai/maxence-frenette/chess-engine-4/runs/hzn4nw8w |
| `1e16` | `d192x6h6` | 1024 | 0.0003 | 1,555,456 | 5.1914 | 0.2379 | 958.5s | https://wandb.ai/maxence-frenette/chess-engine-4/runs/0l1y6qut |

The report was generated separately from the MLP report under `reports/scaling-laws-transformer64/`.
