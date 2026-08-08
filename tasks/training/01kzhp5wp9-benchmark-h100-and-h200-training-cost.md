---
id: "01kzhp5wp9"
title: "Benchmark H100 and H200 training cost"
status: pending
priority: high
effort: medium
dependencies: ["01kzhp5w47"]
tags: ["sm90", "modal", "benchmark", "cost"]
created_at: 2026-08-08
---

# Benchmark H100 and H200 training cost

## Objective

Measure whether H100 or H200 is the cheapest supported GPU for any canonical
model using steady-state end-to-end training cost, then update the checked-in
per-width recipe selections manually.

## Tasks

- [ ] Profile H100 and H200 against every other supported GPU for each relevant
      canonical model configuration.
- [ ] Hold model, batch, precision, backend, input pipeline, and loader settings
      fixed within each comparison.
- [ ] Compute cost per step from measured wall time and the configured GPU plus
      reserved CPU rates; exclude startup cost.
- [ ] Record the measurements and manually update `configs/dense.py` and
      `configs/moe64a2.py` only where a new GPU is cheaper.

## Acceptance Criteria

- Every supported H100/H200 comparison has retained, reproducible throughput
  evidence or an explicit incompatibility result.
- Cost calculations include reserved CPU charges and identify the pricing
  snapshot used.
- Canonical GPU choices remain literal checked-in recipe data, while `--gpu`
  remains an arbitrary experiment override.
- The launch summary and policy documentation agree with the selected recipes.
