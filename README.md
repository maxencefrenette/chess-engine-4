# Chess Engine 4

Chess Engine 4 is a compact Python training codebase for an LCZero-compatible
neural network. The initial shape is intentionally monolithic: one package owns
data loading, training entrypoints, and eventually model export.

## Prerequisites

- [mise](https://mise.jdx.dev/)

```sh
uv sync --dev
```

## Data

The neural net is trained on lc0 t80 data. Set the data path in `.env`.

## Scripts

`uv run train` trains the MLP-only model and logs metrics to the W&B project
configured in `.env`.

```sh
uv run train --config configs/1e14.toml
```

For local dry runs without W&B:

```sh
uv run train --config configs/1e14.toml --no-wandb --batch-size 4 --flops-target 1e9 --device cpu
```

To launch the same training loop on Modal:

```sh
uv run train-modal --config configs/1e14.toml
```

Set W&B configuration with environment variables such as `WANDB_PROJECT`,
`WANDB_ENTITY`, and `WANDB_MODE`.

Training runs are defined by TOML files under `configs/`. The training budget is
`flops_target`; the trainer measures FLOPs per sample with PyTorch and computes
the step count from that budget. Environment variables own local paths and W&B
configuration; the CLI is for invocation-time overrides such as the config path,
FLOPs target, batch size, model width/depth, device, W&B on/off, and W&B run
name.

# Methodology

Hyperparameters are swept at smaller FLOPs budgets, and the best runs are used to fit scaling laws in order to extrapolate the best hyperparameters for the next FLOPs budget. Due to some FLOPs budget being quite small and noisy, all end-of-run metrics are averaged over the last 100 steps of training.

# Scaling Laws

To fit scaling laws and extrapolate the next FLOPs budget from the current best
W&B runs:

```sh
uv run scaling-laws --target-flops 1e16
```

This also writes a Markdown report and SVG charts under `reports/scaling-laws/`.
