---
title: "Support local model training"
id: "01m253ns0j"
status: completed
priority: high
type: feature
tags: ["training", "local", "cuda"]
created_at: "2026-09-10"
completed_at: 2026-09-10
---

# Support local model training

## Objective

Make the production CUDA training loop directly runnable on a supported local
GPU, with Modal reduced to infrastructure and serialization around the same
local runtime.

## Tasks

- [x] Extract shared launch argument, configuration, and reporting code from the
  Modal module.
- [x] Add an RTX 5070 local hardware identity and a `train-local` command.
- [x] Keep Modal training as a thin wrapper over the shared training runtime.
- [x] Document local CUDA setup, data selection, checkpoints, and smoke tests.
- [x] Convert and audit a small local Parquet shard.

## Progress

- 2026-09-10: Converted
  `training-run1-test80-20240428-1817.tar` into a 612,142-byte local Parquet
  shard containing 6,948 positions at `/data/chess/t80/parquet-sample/`.
- 2026-09-10: Installed the CUDA extra on the RTX 5070 and completed a one-step
  BF16 d64 smoke run with batch 128. It processed 128 positions at 19.7
  positions/s, ended at loss 8.1844, and wrote the final checkpoint under
  `artifacts/local-smoke/`.

## Acceptance Criteria

- `train-local` resolves the actual GPU, reads local Parquet, and writes a final
  checkpoint without importing Modal orchestration.
- `train-modal` retains its launch summary, payload, volume commits, and result
  reporting while calling the same training runtime.
- A one-step local RTX 5070 smoke test completes against verified converted data.
- Focused tests and repository-wide Python verification pass.
