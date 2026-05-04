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
uv run train-modal --config configs/1e14.toml --gpu l4
```

Set W&B configuration with environment variables such as `WANDB_PROJECT`,
`WANDB_ENTITY`, and `WANDB_MODE`.

Training runs are defined by TOML files under `configs/`. The training budget is
`flops_target`; the trainer measures FLOPs per sample with PyTorch and computes
the step count from that budget. Environment variables own local paths and W&B
configuration; the CLI is for invocation-time overrides such as the config path,
FLOPs target, batch size, device, W&B on/off, and W&B run name.

Modal uses the `chess-engine-4-training-data` Volume mounted at
`/data/training_data` and the `chess-engine-4-wandb` Secret for W&B environment
variables.
