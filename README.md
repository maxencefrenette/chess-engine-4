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

## Optimization

See [OPTIMIZATION.md](OPTIMIZATION.md) for the compute-budget convention and the
config rules used when tuning model quality versus run cost.

## Prerequisites

- [mise](https://mise.jdx.dev/)

```sh
uv sync --dev
```

## Running Training

Set appropriate environment variables in `.env`. See `.env.example`.

For local training, build the Rust dataloader extension once:

```sh
uv run maturin develop --manifest-path crates/leela_loader/Cargo.toml --release
```

Training requires the native dataloader. Modal builds it into the training image
automatically.

`uv run train` trains the model selected by `[model].kind` and logs metrics to
the W&B project configured in `.env`.

```sh
uv run train --config configs/mlp/1e18.toml
```

To train the starter Transformer64 config:

```sh
uv run train --config configs/transformer64/1e14.toml
```

For local dry runs without W&B:

```sh
uv run train --config configs/mlp/1e18.toml --no-wandb --batch-size 4 --compute-budget 1e9 --device cpu
```

To save a final local checkpoint:

```sh
uv run train --config configs/mlp/1e18.toml --checkpoint-dir checkpoints
```

To launch the same training loop on Modal:

```sh
uv run train-modal --config configs/mlp/1e18.toml
```

Modal training uses the GPU type from the model config by default. CUDA training is hardcoded to bf16 and requires a bf16-capable GPU such as L4 or newer; local CPU/MPS runs use fp32.

To profile the Modal training loop:

```sh
uv run profile-training --config configs/mlp/1e18.toml
```

To save Modal checkpoints into the `chess-engine-4-artifacts` Volume:

```sh
uv run train-modal --config configs/mlp/1e18.toml --save-checkpoints
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
MLP W&B runs:

```sh
uv run scaling-laws --target-compute-budget 1e16
```

This also writes a Markdown report and SVG charts under `reports/scaling-laws/mlp/`.

For Transformer64, use the matching best-run file and report root:

```sh
uv run scaling-laws --best-runs experiments/best-runs-transformer64.toml --output-root reports/scaling-laws/transformer64 --config configs/transformer64/1e16.toml --target-compute-budget 1e17
```
