# Optimization Notes

## Experiment Rules

Keep the loss configuration fixed unless the training target itself is under
study. Model-quality experiments may change the model, run, and optimizer.
Infrastructure settings should only reduce cost or wall-clock time for an
otherwise identical run.

Compare completed runs with `uv run compare-run WANDB_URL`. The command fits loss
against actual training FLOPs from the current best runs and reports `EG_flops`:

```text
EG_flops = fitted FLOPs required for the observed loss / actual training FLOPs
```

Values above `1x` beat the fitted frontier. At an existing width, promote a run
when its `EG_flops` exceeds the incumbent run's value. Runs with detected loss
spikes remain invalid.

## Transformer Engine

Training is CUDA-only and uses NVIDIA Transformer Engine for MLP layers. The
packed input and model are captured with TE's high-level CUDA graph API.

`infra.cpu_cores` controls the physical CPU cores reserved by Modal, while
`infra.dataloader_threads` controls the Rust batch-loading workers. Baseline
configs reserve eight cores and use eight workers; larger runs can reserve more
CPU headroom without changing the data-loading topology.
