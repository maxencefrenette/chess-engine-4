# Chess Engine 4

Chess Engine 4 trains LCZero-compatible neural networks on Modal. The goal is
to create strong nets that run directly inside lc0 through project-owned CUDA
kernels.

The repository is intentionally monolithic: model definitions, training,
datasets, evaluation, experiments, custom kernels, and the scaling dashboard
live together.

## Models

- `dense`: stacked MLP layers over flattened LCZero input planes.
- `moe64a2`: alternating dense and 64-expert layers with two active experts.

Both families predict LCZero policy logits, WDL value logits, and moves left.
Training is Blackwell-only and uses NVIDIA Transformer Engine. Training data is
supervised LCZero t80 data stored as Parquet.

## Setup

Install [mise](https://mise.jdx.dev/), then prepare the Python environment:

```sh
mise install
uv sync --dev
```

Configure W&B and data paths in `.env`; see `.env.example` for the supported
variables.

## Training

Run the canonical dense recipe on Modal:

```sh
uv run train-modal --d-model 128
```

Run the MoE recipe with:

```sh
uv run train-modal --config configs/moe64a2.py --d-model 128
```

The Python recipes derive the complete training configuration from model width
and training ratio. CLI flags are reserved for controlled experiment overrides.
Training always saves periodic and final checkpoints to the Modal artifacts
volume.

See [OPTIMIZATION.md](OPTIMIZATION.md) for the experiment-selection protocol.

## Data

Build the local Rust loader and inspect the configured Parquet dataset:

```sh
uv run maturin develop --manifest-path crates/leela_loader/Cargo.toml --release
uv run inspect-data
```

Convert a local LCZero archive with:

```sh
uv run lc0-to-parquet training.tar training.parquet
```

## Website

Launch the dashboard and its scaling-data watcher together:

```sh
uv run website
```

Open [http://localhost:3000](http://localhost:3000). Changes to recipes,
best-run registries, throughput measurements, and scaling code update the site
automatically.

## Commands

| Command | Purpose |
| --- | --- |
| `uv run train-modal` | Train a model on Modal |
| `uv run profile-training` | Profile the production training loop |
| `uv run throughput-sweep` | Refresh cached throughput measurements |
| `uv run benchmark-training-modal` | Compare Transformer Engine and custom kernels |
| `uv run export-model` | Export a BF16 dense checkpoint as Safetensors |
| `uv run build-lc0` | Build the vendored lc0 fork and custom backend |
| `uv run benchmark-lc0-modal` | Benchmark the custom lc0 backend on RTX PRO 6000 |
| `uv run eval-modal` | Evaluate an exported net with lc0 and fastchess |
| `uv run eval-tournament-modal` | Run an adaptive, resumable Elo tournament |
| `uv run compare-run` | Compare a W&B run with the training frontier |

Use `--help` on a command for its options. Custom-kernel development is
documented in [kernels/README.md](kernels/README.md); experiment reports live in
[`experiments/`](experiments/).

The lc0 fork is maintained as a git subtree in `third_party/lc0`. Update it with
`git subtree pull --prefix=third_party/lc0 https://github.com/LeelaChessZero/lc0.git master --squash`.

## Verification

```sh
uv run pytest -q
uv run ruff check .
pnpm --dir website lint
pnpm --dir website build
```
