---
id: "01kzg0rdp1"
title: "Validate NVFP4 training"
status: pending
priority: low
effort: large
dependencies: []
tags: ["precision", "kernels", "notion-import"]
touches: ["training", "kernels"]
created_at: 2026-08-07
---

# Validate NVFP4 training

## Objective

Establish whether NVFP4 can train useful dense or MoE networks on supported
Blackwell shapes, and quantify its realized cost advantage and numerical risk
against the current BF16 or MXFP8 recipe.

## Tasks

- [ ] Identify model shapes where NVFP4 kernels are technically appropriate.
- [ ] Validate forward, backward, optimizer, checkpoint, and export behavior.
- [ ] Run matched short throughput and numerical-correctness benchmarks.
- [ ] Run a controlled training comparison where the smoke tests pass.

## Acceptance Criteria

- Reports end-to-end throughput, realized dollar cost, and final loss tradeoffs.
- Defines numerical acceptance thresholds and records any unsupported shapes.
- Does not promote NVFP4 solely from a microbenchmark win.

## Source

[Lc0 Net](https://app.notion.com/p/35a8054d223580b79ebadc55321dd4d3)
