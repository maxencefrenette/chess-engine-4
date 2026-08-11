---
id: "01kzg0rdp1"
title: "Validate NVFP4 dense training"
status: completed
priority: high
effort: small
dependencies: []
tags: ["precision", "kernels", "notion-import"]
touches: ["training", "kernels"]
created_at: 2026-08-07
completed_at: 2026-08-11
---

# Validate NVFP4 dense training

## Objective

Decide whether NVFP4 warrants dense training work at relevant model widths.

## Tasks

- [x] Benchmark paired MXFP8 and NVFP4 inference across large widths.
- [x] Locate the performance crossover and record the decision.

## Acceptance Criteria

- The paired benchmark and crossover are retained in an experiment report.
- Further work is stopped if NVFP4 is not useful at relevant widths.

## Outcome

NVFP4 was 0.728-0.771x as fast as MXFP8 at d2048 and crossed over only
between d2048 and d3072. Those widths exceed current needs, so training and
numerical validation are not worth pursuing now. See
`experiments/2026-08-11.03-nvfp4-inference-crossover/`.

## Source

[Lc0 Net](https://app.notion.com/p/35a8054d223580b79ebadc55321dd4d3)
