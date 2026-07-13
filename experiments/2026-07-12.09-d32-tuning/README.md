# d32 Tuning

## Goal

Establish a small `d32` SwiGLU model below the existing `d64` frontier point.
Selection uses `loss_upper_1sd` versus modified compute because the experiment
jointly tunes batch allocation. Modified compute is
`flops_per_sample * batch_size * steps^2`.

All runs used 7,500 steps, 4x expansion, MXFP8, the canonical loss recipe, and
the same training data. The first ten runs were the planned budget; three extra
depth-3 checks tested the strongest allocation after the initial shape decision.

## Shape Pass

| Depth | Batch | LR | Loss | Loss + 1 SD | Modified efficiency | W&B |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1,536 | 1.6e-3 | 3.8471 | 4.0263 | 0.888x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/5xp01ijj) |
| 2 | 1,536 | 1.6e-3 | 3.8325 | 4.0125 | 0.926x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/7hb460tb) |
| 3 | 1,536 | 1.6e-3 | 3.8282 | **3.9949** | **0.990x** | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/2uey3pqg) |
| 4 | 1,536 | 1.6e-3 | 3.8293 | 4.0092 | 0.879x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/rolrvuwd) |

Depth 3 won the initial shape pass on the relevant modified-compute metric.

## Batch and Learning Rate

The first grid used depth 2 because physical-FLOPs efficiency was initially
applied during the shape pass. The selection was corrected to modified-compute
efficiency before choosing the final recipe.

| Depth | Batch | LR | Loss | Loss + 1 SD | Modified efficiency | W&B |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 1,536 | 1.3e-3 | 3.8440 | 4.0266 | 0.854x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/zrew9bqk) |
| 2 | 1,536 | 1.7e-3 | 3.8309 | 4.0108 | 0.935x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/vv5hv3vx) |
| 2 | 2,048 | 1.3e-3 | 3.8001 | 3.9686 | 0.900x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/ywdqhm9a) |
| 2 | 2,048 | 1.7e-3 | **3.7926** | **3.9508** | **1.002x** | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/l0qiw2dj) |
| 2 | 3,072 | 1.3e-3 | 3.7914 | 3.9296 | 0.760x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/bhfwsxan) |
| 2 | 3,072 | 1.7e-3 | 3.7721 | 3.9123 | 0.845x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/vv1jw9v5) |

## Depth-3 Verification

| Depth | Batch | LR | Loss | Loss + 1 SD | Modified efficiency | W&B |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 2,048 | 1.5e-3 | 3.7988 | 3.9594 | 0.917x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/tm0bg66g) |
| 3 | 2,048 | 1.7e-3 | **3.7947** | **3.9531** | **0.952x** | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/qz1ib5tm) |
| 3 | 2,048 | 1.9e-3 | 3.7968 | 3.9543 | 0.946x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/0l605h96) |

Depth 3 did not match the depth-2 winner at the improved allocation.

## Selected Recipe

```toml
[run]
steps = 7500
batch_size = 2048

[model]
d_model = 32
depth = 2

[optimizer]
lr = 1.7e-3
```

The selected run has 318,496 parameters, sees 15,360,000 samples, and uses
`2.312e17` modified compute. It scored 1.002x against the pre-promotion curve
and 1.001x after refitting the curve with `d32` included. It becomes the
canonical `d32` point.
