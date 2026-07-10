# Chess Engine 4

Chess Engine 4 is a compact Python training codebase for an LCZero-compatible
neural network.

The goal is to create strong nets that can be run directly inside lc0 through
the ONNX backend.

## Models

The training loop supports multiple model kinds behind one output contract:
LCZero-style policy logits, WDL value logits, and a moves-left prediction.

The dense MLP model is a stack of pre-norm SwiGLU blocks over flattened LCZero
input planes. The MoE family replaces each dense MLP block with a routed
SwiGLU mixture of experts. Both training models use NVIDIA Transformer Engine;
the MoE path uses its fused grouped-linear operation pipeline.

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

To inspect training data locally, build the Rust dataloader extension once:

```sh
uv run maturin develop --manifest-path crates/leela_loader/Cargo.toml --release
uv run inspect-data
```

By default, `inspect-data` validates one batch. Pass `--batches N` for a bounded
scan or `--all` to inspect every batch.

Training runs on Modal. Modal builds the native dataloader into its image
automatically. `uv run train-modal` trains the model selected by `[model].kind`
and logs metrics to the W&B project configured in `.env`.

```sh
uv run train-modal --config configs/mlp/1e18.toml
```

Modal training uses the GPU type from the model config by default. Training is
CUDA-only, hardcoded to bf16, and requires a bf16-capable GPU such as L4 or
newer. The Modal image builds the pinned Transformer Engine version and its
PyTorch extension automatically. Training functions reserve eight CPU cores
for the background Rust dataloader.

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

Conversion reconstructs a portable inference model from the Transformer Engine
checkpoint. The resulting ONNX graph has no Transformer Engine dependency and
keeps MoE top-k routing sparse by evaluating only the selected experts.

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

The command prints the fitted laws, observed runs, extrapolated target, and launch
command. Use `--write-config PATH` to write the suggested TOML configuration.
