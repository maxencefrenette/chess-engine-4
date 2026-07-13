# d288 Activation Sweep

## Goal

Repeat the activation comparison at `d288x5`, where residual MLP blocks make up
a larger share of total computation. All runs used 4x expansion, batch size
16,384, 11,448 steps, learning rate 7.5e-4, the same training data and optimizer,
and MXFP8 precision.

## Results

| Activation | Training FLOPs | Loss | Loss + 1 SD | Policy top-1 | FLOPs efficiency | Modified-compute efficiency | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| GEGLU | 8.751e15 | **3.1363** | **3.1818** | 41.79% | **1.226x** | **1.202x** | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/4g4vkfr7) |
| SwiGLU | 8.737e15 | 3.1452 | 3.1921 | **41.89%** | 1.110x | 1.072x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/fr7vpym5) |
| GELU | 6.881e15 | 3.1860 | 3.2293 | 40.34% | 0.896x | 0.905x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/7wzwxixl) |
| SiLU | 6.867e15 | 3.2213 | 3.2693 | 39.42% | 0.614x | 0.595x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/f5bk9vjm) |
| ReLU squared (`srelu`) | 6.870e15 | 3.2306 | 3.2768 | 38.92% | 0.556x | 0.551x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/6krhm2xm) |

The non-gated variants used about 21.4% fewer total FLOPs than SwiGLU, compared
with about 13% at `d128`. The saving grew as expected, but none recovered enough
quality to become compute-efficient.

## Selection

GEGLU improved physical-FLOPs efficiency from the strongest existing `d288`
incumbent's 1.123x to 1.226x, a gain of 0.103x. It also had the best mean loss
and uncertainty-adjusted loss in this sweep. Its policy top-1 was 0.10 percentage
points below the fresh SwiGLU control, but loss is the training selection metric.

GEGLU therefore replaces SwiGLU in `configs/dense/d288.toml` and this run becomes
the canonical `d288` entry in `experiments/best-runs-dense.toml`.

## Commands

```bash
uv run train-modal --config configs/dense/d288.toml --activation swiglu --wandb-name activation-d288-swiglu
uv run train-modal --config configs/dense/d288.toml --activation geglu --wandb-name activation-d288-geglu
uv run train-modal --config configs/dense/d288.toml --activation gelu --wandb-name activation-d288-gelu
uv run train-modal --config configs/dense/d288.toml --activation silu --wandb-name activation-d288-silu
uv run train-modal --config configs/dense/d288.toml --activation srelu --wandb-name activation-d288-srelu
```

All five jobs ran concurrently on B200s.
