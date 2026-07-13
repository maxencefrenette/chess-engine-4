# Dense Family Baseline

Commit: `a659480`

This establishes the first baseline generated entirely by `configs/dense.py`.
The four widths ran concurrently with no CLI hyperparameter overrides and no
detected loss spikes.

| Model | Command | W&B | Loss | Loss + 1 SD | Policy top-1 | FLOPs efficiency | Old width verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `d32x2` | `uv run train-modal --d-model 32` | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/72c3f49e) | 3.7908 | 3.9538 | 0.2533 | 0.996x | Promote |
| `d64x3` | `uv run train-modal --d-model 64` | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/bzyu2wem) | 3.5664 | 3.6576 | 0.3047 | 0.986x | Keep incumbent |
| `d128x4` | `uv run train-modal --d-model 128` | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/ws5uysef) | 3.3536 | 3.4113 | 0.3605 | 0.991x | Keep incumbent |
| `d256x5` | `uv run train-modal --d-model 256` | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/d4u870kc) | 3.1349 | 3.1750 | 0.4182 | 1.228x | Promote |

The old `d32` through `d288` points were deleted and replaced despite the `d64`
and `d128` width verdicts. This is a methodology reset: active low-scale points
now follow one smooth family recipe instead of mixing independently tuned model
shapes and allocations. The more expensive `d576`, `d896`, and `d1472` runs are
retained as stale history until the new recipe can be run at those scales.
