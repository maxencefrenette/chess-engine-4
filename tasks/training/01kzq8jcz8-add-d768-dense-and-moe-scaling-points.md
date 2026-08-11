---
id: "01kzq8jcz8"
title: "Add d768, d1280, and d1536 scaling points"
status: pending
priority: high
effort: large
dependencies: []
tags: ["training", "scaling", "dense", "moe", "kernels", "benchmark"]
created_at: 2026-08-10
---

# Add d768, d1280, and d1536 scaling points

## Objective

Add d768, d1280, and d1536 to the dense and MoE64A2 ladders, and remove dense
d2048. Use the cheapest stable GPU, precision, and kernel path for each point.

## Tasks

- [ ] Update both ladders for the three widths and remove dense d2048.
- [ ] Extend kernels and strict dispatch; verify numerics and neighboring shapes.
- [ ] Benchmark viable GPUs and Transformer Engine/custom paths end to end.
- [ ] Select batch, precision, backend, and GPU by steady-state training cost.
- [ ] Train dense at 0.2x and MoE64A2 at 0.05x; run `compare-run`.

## Acceptance Criteria

- All new widths pass existing numerical and dispatch thresholds.
- Dense d2048 is absent from the recipe and canonical ladder metadata.
- Benchmarks include loader, CPU charges, throughput, memory, and pricing basis.
- Dense and MoE points use independently measured cheapest stable recipes.
- Promotable runs update canonical best-runs and generated website data.
