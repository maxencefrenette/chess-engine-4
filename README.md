# Chess Engine 4

Train chess networks on the local GPU and publish them for **our custom lc0
engine**, using its `ce4` backend. Stock-lc0 model compatibility is not a target.

The main workflow is: select data → train locally → evaluate → export and publish.
Modal is optional, for hardware or data unavailable locally. Research planners
and the public experiment dashboard are separate from the training runtime.

## Local setup

Toolchains are managed by mise; Python dependencies by uv:

```sh
mise install
uv sync --dev
mise run setup-cuda
```

The CUDA setup installs Transformer Engine and builds the Rust data loader.
It is a substantial installation/compilation step, not a routine check. For
CPU tests and analysis only, see [development setup](docs/development.md).

Configure W&B and your data path in `.env`; see [.env.example](.env.example).
Local training does not require the Modal SDK or a Modal account.

## Train

```sh
uv run train-local --d-model 128 --data /path/to/parquet
```

`configs/dense.py` is the canonical dense recipe. Recipes derive model and
training settings from width and training ratio; routine experiments default
to `0.2x` Chinchilla. `--config configs/moe64a2.py` selects the MoE family.

The launcher detects the local GPU, checks architecture compatibility, and
prints the launch summary. Add `--dry-run` to inspect the resolved configuration
without training. The canonical batches came from cloud experiments: architecture
compatibility does **not** guarantee a recipe fits local VRAM. Review the batch
size and training budget before launching; local throughput tuning is separate work.

Periodic and final checkpoints are written under `checkpoints/`. Use
`--checkpoint-dir` and `--checkpoint-every` to change that, and `--no-wandb`
for a smoke test. Checkpoint saving exists, but training resume is not implemented.

### Data readiness

The current reader supports the older training Parquet schema. The local T80
Iceberg conversion is complete (43,802,607,232 positions), but reader integration
and reproducible subset/full-corpus mixing are still pending. Do not point
`train-local --data` directly at the Iceberg warehouse yet.

See [data formats and maintenance](docs/data.md) for corpus locations, metadata,
and conversion tools.

## Evaluate and publish for our engine

Compare a completed run with the training frontier:

```sh
uv run compare-run WANDB_URL
```

Export a trusted local checkpoint into the custom engine's Safetensors format:

```sh
uv run export-model checkpoints/RUN-final.pt \
  --output artifacts/models/RUN.safetensors
```

Local export runs on CPU without Modal. A missing path is an error; remote
export requires [an explicit `--remote` flag](docs/cloud.md#remote-export-is-explicit).
Current export supports the dense and `moe64a2` SwiGLU models with BF16 weights.
Only load checkpoints you trust: the training checkpoint loader uses pickle.

Build our engine with `uv run build-lc0` on a CUDA build host. The exported
weights require this engine's `ce4` backend, not a stock lc0 binary.
Keep the engine revision and supported GPU architecture with each model release.

Evaluation orchestration currently runs through [Modal commands](docs/cloud.md).
Reusable pairing, result parsing, and Elo fitting live in `evaluation/`;
a local evaluation launcher remains follow-up work.

Before publishing, retain the weights/checksum, exact recipe and code revision,
data selection, engine revision, evaluation evidence, and usage instructions.
Automated release assembly/upload is not implemented yet. The existing
[publishing task](tasks/evaluation/01kzhkmxcz-evaluate-and-publish-the-50-dollar-model.md)
tracks the first release. Experiment promotion rules live in [AGENTS.md](AGENTS.md).

## Repository map

| Location | Responsibility |
| --- | --- |
| `src/chess_engine_4/training/` | Local/shared training loop, configuration, losses, optimizers |
| `src/chess_engine_4/model/` | Model definitions and portable reference/FLOP models |
| `src/chess_engine_4/data/`, `crates/leela_loader/` | Dataset ingestion and native loading |
| `src/chess_engine_4/evaluation/` | Reusable evaluation configuration and analysis |
| `src/chess_engine_4/release/` | Custom-lc0 model export |
| `src/chess_engine_4/backends/modal/` | Optional cloud execution adapters |
| `src/chess_engine_4/analysis/` | Research planning, scaling fits, throughput cache, dashboard generation |
| `kernels/`, `third_party/lc0/` | Training/inference kernels and our engine fork |
| `experiments/` | Preserved results and canonical registries |
| `website/` | Public experiment dashboard |
| `tasks/` | Durable work and backlog |

Both dense and MoE families predict policy, WDL, and moves left. Model numerics,
historical results, and alternate-GPU kernels are retained. This is a package
organization change, not a recipe or architecture change.

## Further documentation

- [Development and verification](docs/development.md)
- [Optional cloud execution](docs/cloud.md)
- [Research analysis and planning](docs/research.md)
- [Training optimization notes](OPTIMIZATION.md)
- [Website development](website/README.md)

Existing CLI names remain available; use `uv run COMMAND --help`. Python module
paths have moved with the package layout, and repository scripts use the new
imports. The only intentional CLI behavior change is explicit remote export.
