---
id: "01kzs2eysd"
title: "Optimize Muon kernels and tune learning rate"
status: completed
priority: medium
effort: medium
dependencies: []
tags: ["optimizer", "kernel", "experiment"]
created_at: 2026-08-11
completed_at: 2026-08-11
---

# Optimize Muon kernels and tune learning rate

## Objective

Reduce Muon's realized training overhead by batching same-shaped hidden matrices,
then lightly tune its learning rate at the widths where the pilot showed the
strongest convergence signal and the first large-scale exception.

## Tasks

- [x] Implement a batched Muon update with PyTorch Muon as the numerical reference.
- [x] Benchmark optimizer-only and canonical end-to-end step time at d256 and d512.
- [x] Run a narrow matched LR comparison at d256 and d512.
- [x] Report `EG_flops`, runtime, realized efficiency, and a promotion verdict.

## Acceptance Criteria

- The optimized update passes numerical parity checks against PyTorch Muon.
- End-to-end measurements use the same model, batch, data, and hardware as the pilot.
- LR selection uses only spike-free runs and does not change the canonical recipe without review.
