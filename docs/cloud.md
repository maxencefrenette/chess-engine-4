# Optional Modal execution

Local training and model export do not require Modal. Install its optional extra
and configure your Modal account only when cloud hardware or data is needed:

```sh
uv sync --dev --extra modal
uv run --extra modal train-modal --d-model 128
```

Use `--extra modal` on cloud commands so uv keeps the optional SDK available.
Cloud CUDA dependencies and native builds are handled by the image definition;
the workstation does not need the local `cuda` extra to submit a cloud job.

## Commands

All existing command names remain available. Their execution adapters live in
`src/chess_engine_4/backends/modal/`.

| Command (prefix with `uv run --extra modal`) | Purpose |
| --- | --- |
| `train-modal` | Run the shared training loop on Modal |
| `profile-training` | Profile the training loop on Modal |
| `throughput-sweep` | Refresh the research throughput cache |
| `benchmark-training-modal` | Compare Transformer Engine and custom kernels |
| `benchmark-moe-kernels-modal` | Check and benchmark custom MoE kernels |
| `prepare-lc0-modal` | Build the custom lc0 engine remotely |
| `benchmark-lc0-modal` | Benchmark engine inference |
| `eval-modal` / `eval-selfplay-modal` | Evaluate exported networks |
| `eval-tournament-modal` | Run an adaptive, resumable Elo tournament |
| `benchmark-tournament-modal` | Benchmark each configured engine |
| `eval-precision-modal` | Compare precision recipes |

Run `COMMAND --help` for arguments. Cloud runs save periodic and final
checkpoints in the Modal artifacts volume. MoE uses the same launcher with
`--config configs/moe64a2.py`.

## Remote export is explicit

```sh
uv run --extra modal export-model checkpoints/RUN-final.pt --remote \
  --output artifacts/models/RUN.safetensors
```

Add `--remote-only` to retain the exported artifact only in the Modal volume.
Without `--remote`, the path is always local and a missing checkpoint is an
error; it never silently falls back to a cloud job. Exported weights target
our custom lc0 `ce4` backend, exactly as local exports do.

## Data maintenance

The older Modal Parquet corpus is separate from the completed local Iceberg
corpus. Uploading the latter does not by itself make it readable by training.
See [data.md](data.md) before changing either corpus.

Cloud maintenance commands are `convert-data-modal`, `verify-data-modal`,
`audit-data-modal`, and `audit-sources-modal`. The scripts
`scripts/sync_modal_training_data.py` and `scripts/audit_modal_parquet_targets.py`
also require the Modal extra. They remain available for maintaining historical
data; they are not part of the local train-to-release workflow.
