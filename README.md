# Chess Engine 4

Chess Engine 4 is a compact Python training codebase for an LCZero-compatible
neural network.

The goal is to create strong nets that can be run directly inside lc0 through
the ONNX backend.

## Model

The `dense` model is a stack of MLP layers over flattened LCZero input
planes with LCZero-style policy logits, WDL value logits, and a moves-left
prediction. Training uses NVIDIA Transformer Engine.

The models are trained on lc0 t80 data using supervised learning.

## Optimization

See [OPTIMIZATION.md](OPTIMIZATION.md) for the protocol used to decide whether an
experiment improves the training frontier.

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
automatically. `uv run train-modal` logs metrics to the W&B project configured
in `.env`. Routine runs default to `0.2x` Chinchilla; pass
`--training-ratio 1` explicitly for a full-training run.

```sh
uv run train-modal --d-model 64
```

`configs/dense.py` is the canonical dense-family recipe. Width and training
ratio are its scaling arguments; the recipe derives depth, batch size, steps,
learning rate, and the remaining training configuration. CLI flags can override
those derived values for controlled experiments. The baseline batch size is
`32 * d_model`. Before submitting work to Modal, training commands print the
fully resolved shape, parameter count, batch size, steps, samples, FLOPs,
precision, and CPU allocation.

Training is Blackwell-only and runs on a Modal B200. Evaluation may use cheaper
Modal GPUs when its backend supports them. Models use Transformer Engine MXFP8
block scaling and FP32 optimizer master weights.
Training is CUDA-graphed. The Modal image builds the pinned Transformer Engine
version and its PyTorch extension automatically. Training reserves eight CPU
cores for the background Rust dataloader.

MXFP8 projections are padded to 32-element boundaries and sliced back to the
LC0 output contract.

The recipe's precision setting accepts `bf16`, `mxfp8`, or `nvfp4`. It can be
overridden for profiling with `--quantization-recipe`.

To profile the Modal training loop:

```sh
uv run profile-training --d-model 64
```

To benchmark the canonical dense ladder concurrently and cache the results in
`experiments/throughput-dense.toml`:

```sh
uv run throughput-sweep
```

Matching cached widths are skipped. Pass `--refresh` to replace them or
`--widths 256 512 1024` to benchmark a subset.

Modal training always writes checkpoints to the `chess-engine-4-artifacts`
Volume every 50,000 steps and at the end of the run:

```sh
uv run train-modal --d-model 64
```

To export a saved Modal checkpoint with Transformer Engine and package it as an
lc0 ONNX weights file:

```sh
uv run checkpoint2leela checkpoints/run-final.pt --output artifacts/leela/run.pb.gz
```

Export runs on a Modal B200 using Transformer Engine's native ONNX export context.
It writes an FP32 graph for lc0's TensorRT-backed `onnx-trt` backend by default.
Use `--export-dtype fp16` for a smaller experimental artifact; FP16 was weaker in
paired engine testing. ONNX Runtime does not register Transformer Engine's
MXFP8-specific TensorRT operators.

To run a small lc0-vs-BT4 match on Modal with fastchess:

```sh
uv run prepare-lc0-modal
uv run eval-modal artifacts/leela/run.pb.gz --nodes 64 --games 2 --rounds 1
```

Evaluation always uses the official Stockfish `noob_2moves.epd` book with a
fixed random seed. Fastchess repeats each selected opening with colors reversed.

`prepare-lc0-modal` builds the Linux CUDA/ONNX lc0 binary once on Modal and
caches it under `/artifacts/bin/lc0`. The eval command uploads the candidate
weights, downloads the prebuilt fastchess release into the Modal image, uses the
cached lc0 binary, and writes PGNs under `/artifacts/evals/<name>/games.pgn`.

To compare a checkpoint's native Transformer Engine outputs with one or more
lc0 ONNX exports over `1,000` training positions:

```sh
uv run eval-inference-modal checkpoints/run-final.pt \
  leela/run-fp32.pb.gz leela/run-fp16.pb.gz
```

The command preserves the eight-position input history, evaluates each export
through lc0 at one node, and stores its JSON summary under
`/artifacts/evals/inference-mismatch/`.

Set W&B configuration with environment variables such as `WANDB_PROJECT`,
`WANDB_ENTITY`, and `WANDB_MODE`.

## Website

The static dashboard derives its curves and next-width projection from the
canonical best-runs registry and family recipe. Generate its input explicitly
with:

```sh
uv run export-scaling-data
```

The website development and production build commands regenerate this data
automatically.
