---
id: "01kzg0rdr7"
title: "Evaluate the Muon optimizer"
status: completed
priority: medium
effort: medium
dependencies: []
tags: ["optimizer", "experiment", "notion-import"]
touches: ["training", "experiments"]
created_at: 2026-08-07
completed_at: 2026-08-11
---

# Evaluate the Muon optimizer

## Objective

Test whether Muon improves training-FLOP or realized-cost efficiency for the
stacked MLP architecture relative to fused AdamW.

## Tasks

- [x] Define which matrix parameters use Muon and how remaining parameters are optimized.
- [x] Benchmark optimizer overhead and memory use.
- [x] Tune the minimum necessary optimizer hyperparameters at a cheap width.
- [x] Run a matched controlled experiment and report EG_flops and realized cost.

## Acceptance Criteria

- The comparison uses the same data, model shape, and training target.
- The report separates convergence gains from optimizer runtime overhead.

## Source

[Lc0 Net](https://app.notion.com/p/35a8054d223580b79ebadc55321dd4d3)
