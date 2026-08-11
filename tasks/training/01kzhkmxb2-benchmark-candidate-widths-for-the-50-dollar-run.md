---
id: "01kzhkmxb2"
title: "Benchmark deferred d1536 candidate widths"
status: pending
priority: low
effort: medium
dependencies: []
tags: ["training", "scaling", "benchmark"]
created_at: 2026-08-08
---

# Benchmark deferred d1536 candidate widths

## Objective

Revisit dense d1536 only after the `$50` milestone. MoE d1536 is deferred with
the rest of MoE work. This does not block the final run.

## Tasks

- [ ] Benchmark dense d1536 on the current stack.
- [ ] Compare dense d1024/d1536 loss, parameters, throughput, and cost.

## Acceptance Criteria

- Results use same-stack dense measurements.
- Unsupported shapes fail explicitly; this task does not launch final training.
