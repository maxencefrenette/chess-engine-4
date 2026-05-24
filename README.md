# Chess Engine 4

Chess Engine 4 is a compact Python training codebase for an LCZero-compatible
neural network.

The goal is to create strong nets that can be run directly inside lc0 through
the ONNX backend.

## Models

The training loop supports multiple model kinds behind one output contract:
LCZero-style policy logits, WDL value logits, and a moves-left prediction.

The current MLP model is a stack of pre-norm SwiGLU blocks over flattened LCZero
input planes. Transformer64 is a vanilla attention model with one token per
board square, learned square embeddings, pooled value and moves-left heads, and
an LC0-style attention policy head.

The models are trained on lc0 t80 data using supervised learning.

## Methodology

Hyperparameters are swept at smaller compute budgets, and the best runs are used to fit scaling laws in order to extrapolate the best hyperparameters for larger compute budgets. Due to some compute budgets being quite small and noisy, end-of-run metrics are averaged over the last 100 steps of training.

In order to steer the optimization towards recipes with fewer steps, an alternative compute budget C_eff is defined as:

```math
C_{\text{eff}} = \text{flops\_per\_sample} \times \text{batch\_size} \times \text{steps}^k
```

The default is k = 1.0, which is ordinary physical FLOPs. Setting k > 1.0 turns `compute_budget` into the step-adjusted budget and acts as a soft penalty on steps.

## Prerequisites

- [mise](https://mise.jdx.dev/)

```sh
uv sync --dev
```

## Running Training

Set appropriate environment variables in `.env`. See `.env.example`.

`uv run train` trains the model selected by `[model].kind` and logs metrics to
the W&B project configured in `.env`.

```sh
uv run train --config configs/1e15.toml
```

To train the starter Transformer64 config:

```sh
uv run train --config configs/transformer64/1e14.toml
```

For local dry runs without W&B:

```sh
uv run train --config configs/1e15.toml --no-wandb --batch-size 4 --compute-budget 1e9 --device cpu
```

To save a final local checkpoint:

```sh
uv run train --config configs/1e15.toml --checkpoint-dir checkpoints
```

To launch the same training loop on Modal:

```sh
uv run train-modal --config configs/1e15.toml
```

To save Modal checkpoints into the `chess-engine-4-artifacts` Volume:

```sh
uv run train-modal --config configs/1e15.toml --save-checkpoints
```

To convert a saved checkpoint into an lc0 ONNX weights file:

```sh
uv run checkpoint2leela checkpoints/run-final.pt --output artifacts/leela/run.pb.gz
```

To run a small lc0-vs-BT4 match on Modal with fastchess:

```sh
uv run prepare-lc0-modal
uv run eval-modal artifacts/leela/run.pb.gz --gpu l4 --nodes 64 --games 2 --rounds 1
```

`prepare-lc0-modal` builds the Linux CUDA/ONNX lc0 binary once on Modal and
caches it under `/artifacts/bin/lc0`. The eval command uploads the candidate
weights, downloads the prebuilt fastchess release into the Modal image, uses the
cached lc0 binary, and writes PGNs under `/artifacts/evals/<name>/games.pgn`.

Set W&B configuration with environment variables such as `WANDB_PROJECT`,
`WANDB_ENTITY`, and `WANDB_MODE`.

# Scaling Laws

To fit scaling laws and extrapolate the next compute budget from the current best
W&B runs:

```sh
uv run scaling-laws --target-compute-budget 1e16
```

This also writes a Markdown report and SVG charts under `reports/scaling-laws/`.
