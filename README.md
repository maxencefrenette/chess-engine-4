# Chess Engine 4

Chess Engine 4 trains LCZero-compatible neural networks locally or on Modal. The goal is
to create strong nets that run directly inside lc0 through project-owned CUDA
kernels.

The repository is intentionally monolithic: model definitions, training,
datasets, evaluation, experiments, custom kernels, and the scaling dashboard
live together.

## Models

- `dense`: stacked MLP layers over flattened LCZero input planes.
- `moe64a2`: alternating dense and 64-expert layers with two active experts.

Both families predict LCZero policy logits, WDL value logits, and moves left.
Training is CUDA-only, with Blackwell canonical recipes and BF16 Ampere support.
Training data is supervised LCZero t80 data stored as Parquet.

## Setup

Install [mise](https://mise.jdx.dev/), then prepare the Python environment:

```sh
mise install
uv sync --dev
```

The default environment supports local analysis and planning. The CPU test suite
also exercises the Rust data loader, which must be built separately:

```sh
mise run setup-native
```

Rebuild it after changing the Rust loader or the Python interpreter. This is a
native compilation step and can use substantial CPU; on a shared workstation,
use `CARGO_BUILD_JOBS=1 mise run setup-native` and schedule it around higher-priority
work. It does not require CUDA or Transformer Engine.

CUDA training dependencies are installed explicitly in Modal images. On a compatible
local NVIDIA GPU, install Transformer Engine and rebuild the native loader with:

```sh
mise run setup-cuda
```

Configure W&B and data paths in `.env`; see `.env.example` for the supported
variables.

## Training

Run the canonical dense recipe on the local CUDA GPU:

```sh
uv run train-local --d-model 128 --data /path/to/parquet
```

Local runs save periodic and final checkpoints under `checkpoints/` by default.
Override this with `--checkpoint-dir`, and use `--no-wandb` for smoke tests.
The local launcher detects the physical GPU and rejects configurations meant for
a different architecture. For example, a minimal data-path smoke test is:

```sh
uv run train-local --d-model 64 --steps 1 --batch-size 128 \
  --data /path/to/parquet --no-wandb
```

Run the same recipe on Modal:

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

Plan the predicted lowest-validation-loss dense run with:

```sh
uv run plan-budget 5 10 100
```

Plans default to 25 billion available positions. Use `--assume-samples N` to
evaluate another data budget, or `--dataset experiments/training-data.toml` to
use the recorded corpus size. Add `--families dense moe64a2` to compare both
families. Loss predictions use the curated Skaling `L(N,D)` observations.
Training horizon is capped at twice the largest family-wide observed ratio.
Dense widths may be extrapolated through d2560; their cost holds the largest
measured width's MFU constant and scales wall time by FLOPs per step. Such
costs remain explicitly labeled as width extrapolations.
Predicted losses include deterministic 80% bootstrap intervals. Compare cheap preliminary
runs against spending the same fixed total budget directly on final training with:

```sh
uv run plan-budget 10 --suggest-runs 3
```

Use `--focus-budget` and `--max-suggestion-cost` to change the targeted breakpoint and
per-run cost ceiling. Preliminary-run cost is deducted from the fixed focus budget, and
the preliminary checkpoint is treated as sunk rather than reusable final training.

Scaling-law acquisition uses the declared L-shaped dense grid. Complete its d64
data arm and `0.055x` width arm before ranking cheap, unobserved interior cells:

```sh
uv run plan-ladder --focus-budget 10 --count 3
```

Recorded cells count toward scaffold coverage even when they contain a loss spike;
the spike remains available separately for fit and promotion decisions.

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

There are currently two distinct corpus formats:

| Corpus | Metadata | Status |
| --- | --- | --- |
| Existing Modal Parquet corpus | `experiments/training-data.toml` | 8,020,779,820 positions; supported by the current training reader |
| Local T80 Iceberg table | `experiments/training-data-iceberg.toml` | 43,802,607,232 positions; ingestion complete, training integration pending |

The local Iceberg warehouse is `/data/chess/t80/iceberg`, with table `training.t80`
and a SQLite catalog at `catalog.db`. It is partitioned by
`truncate[4](num_pieces)` and retains source-archive provenance and root, best,
outcome, and moves-left targets. The original LCZero archives were deleted after
the completion audit. See the [ingestion report](experiments/2026-09-17.01-t80-iceberg-ingestion/README.md)
for counts, compression, and omitted corrupt members.

The Iceberg files use different column types from the older Parquet format.
Pointing `train-local --data` at them is not yet supported. Remaining work includes
the reader, reproducible filtering/shuffling, and verified transfer to Modal.
The Iceberg metadata can already supply a planning capacity with
`--dataset experiments/training-data-iceberg.toml`; this does not stage the data
or make a proposed run launchable against Iceberg.

## Website

Install the separate website dependencies using Node.js and the pnpm version
declared in `website/package.json`. For example, with a mise-managed Node 24:

```sh
cd website
mise exec node@24 -- corepack pnpm install --frozen-lockfile
mise exec node@24 -- corepack pnpm dev
```

The website commands also require the root Python environment for generating
dashboard data. They do not require CUDA or the Rust data loader.

Launch the dashboard and its scaling-data watcher together:

```sh
uv run website
```

This convenience launcher requires `pnpm` on `PATH`; the direct Corepack command
above works without that shim. Run root-level commands below from the repository
root.

Open [http://localhost:3000](http://localhost:3000). Changes to recipes,
best-run registries, throughput measurements, and scaling code update the site
automatically.

## Commands

| Command | Purpose |
| --- | --- |
| `uv run train-local` | Train a model on the local CUDA GPU |
| `uv run train-modal` | Train the same model runtime on Modal |
| `uv run profile-training` | Profile the production training loop |
| `uv run throughput-sweep` | Refresh cached throughput measurements |
| `uv run plan-budget 10 100` | Estimate the lowest-loss configuration for dollar budgets |
| `uv run plan-ladder` | Complete the scaling scaffold, then rank cheap in-grid observations |
| `uv run benchmark-training-modal` | Compare Transformer Engine and custom kernels |
| `uv run export-model` | Export a BF16 dense checkpoint as Safetensors |
| `uv run build-lc0` | Build the vendored lc0 fork and custom backend |
| `uv run benchmark-lc0-modal` | Benchmark the custom lc0 backend on a supported GPU |
| `uv run benchmark-tournament-modal` | Benchmark every engine in a tournament config |
| `uv run eval-modal` | Evaluate an exported net with lc0 and fastchess |
| `uv run eval-tournament-modal` | Run an adaptive, resumable Elo tournament |
| `uv run compare-run` | Compare a W&B run with the training frontier |

Use `--help` on a command for its options. Custom-kernel development is
documented in [kernels/README.md](kernels/README.md); experiment reports live in
[`experiments/`](experiments/).

The lc0 fork is maintained as a git subtree in `third_party/lc0`. Update it with
`git subtree pull --prefix=third_party/lc0 https://github.com/LeelaChessZero/lc0.git master --squash`.

## Verification

For quick Python checks without tests, native compilation, or a website build:

```sh
mise run check-fast
```

After `uv sync --dev` and `mise run setup-native`, run the CPU test suite with
`mise run test`. No training or benchmark is launched by these tests. CUDA kernel,
training-throughput, and LCZero inference validation are separate, explicit jobs.

Before committing a broad change, run the full gate from the repository root:

```sh
uv run pytest -q
uv run ruff check .
uv run ty check
pnpm --dir website lint
pnpm --dir website build
git diff --check
```

If `pnpm` is not on `PATH`, run the website checks from `website/` with
`mise exec node@24 -- corepack pnpm lint` and
`mise exec node@24 -- corepack pnpm build` instead. The build regenerates the
dashboard's Python-derived data and is heavier than lint/type checks. On a busy
workstation, run checks sequentially at low priority and limit CPU affinity and
BLAS/OpenMP threads; avoid native rebuilds until CPU time is available.
