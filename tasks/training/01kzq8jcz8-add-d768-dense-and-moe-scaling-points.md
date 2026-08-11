---
id: "01kzq8jcz8"
title: "Add d768 and dense d1280 scaling points"
status: completed
priority: high
effort: large
dependencies: []
tags: ["training", "scaling", "dense", "moe", "kernels", "benchmark"]
created_at: 2026-08-10
completed_at: 2026-08-10
---

# Add d768 and dense d1280 scaling points

## Objective

Add dense d768/d1280 and MoE64A2 d768; remove dense d1536/d2048. Use the
cheapest stable recipe per point. MoE d1280 and both d1536 runs were cancelled.

## Tasks

- [x] Update the revised ladders and remove dense d1536/d2048.
- [x] Extend kernels and strict dispatch; verify numerics and neighboring shapes.
- [x] Benchmark viable GPUs and Transformer Engine/custom paths end to end.
- [x] Select batch, precision, backend, and GPU by steady-state training cost.
- [x] Train two dense points at 0.2x and MoE d768 at 0.05x; run `compare-run`.

## Acceptance Criteria

- All new widths pass existing numerical and dispatch thresholds.
- Dense d2048 is absent from the recipe and canonical ladder metadata.
- Benchmarks include loader, CPU charges, throughput, memory, and pricing basis.
- Dense and MoE points use independently measured cheapest stable recipes.
- Promotable runs update canonical best-runs and generated website data.
