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

Training is CUDA-only and uses NVIDIA Transformer Engine for MLP layers. Dense
models use TE's high-level CUDA graph API.

The `moe64a2` family alternates dense layers with Transformer Engine's fused
top-k router and grouped linear kernels. Widths below 1024 use static
CUDA-graphed dispatch; larger widths use TE's faster fused permutation and
unpermutation path. Total parameters include all 64 experts in each MoE layer;
physical training FLOPs include only the two active experts plus dense and shared
layers. Alternation is a fixed part of the family while small grouped MXFP8
kernels remain expensive.

`infra.cpu_cores` controls the physical CPU cores reserved by Modal, while
`infra.dataloader_threads` controls the Rust batch-loading workers. Baseline
configs reserve eight cores and use eight workers; larger runs can reserve more
CPU headroom without changing the data-loading topology.
