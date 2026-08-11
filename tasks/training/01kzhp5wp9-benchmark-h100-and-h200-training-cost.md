---
id: "01kzhp5wp9"
title: "Benchmark H100 and H200 dense training cost"
status: pending
priority: high
effort: medium
dependencies: ["01kzhp5w47"]
tags: ["sm90", "modal", "benchmark", "cost"]
created_at: 2026-08-08
---

# Benchmark H100 and H200 dense training cost

## Objective

Measure whether H100 or H200 is cheapest for any canonical dense width, then
update the checked-in dense recipe. MoE comparisons are deferred.

## Tasks

- [ ] Profile H100 and H200 against other supported GPUs for each dense width.
- [ ] Hold model, batch, precision, backend, input pipeline, and loader settings
      fixed within each comparison.
- [ ] Compute cost per step from measured wall time and the configured GPU plus
      reserved CPU rates; exclude startup cost.
- [ ] Record measurements and update `configs/dense.py` where a GPU is cheaper.

## Acceptance Criteria

- Every supported H100/H200 comparison has retained, reproducible throughput
  evidence or an explicit incompatibility result.
- Cost calculations include reserved CPU charges and identify the pricing
  snapshot used.
- Canonical GPU choices remain literal checked-in recipe data, while `--gpu`
  remains an arbitrary experiment override.
- The launch summary and policy documentation agree with the selected recipes.
