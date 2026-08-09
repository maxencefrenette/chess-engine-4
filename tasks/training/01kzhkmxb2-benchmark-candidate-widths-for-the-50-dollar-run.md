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

Revisit dense and MoE d1536 only after the current `$50` milestone. Dense d1536
is supported; MoE d1536 remains experimental. This does not block the final run.

## Tasks

- [ ] Benchmark dense d1536 on the current stack.
- [ ] Test MoE d1536 feasibility without making it canonical.
- [ ] Compare d1024/d1536/d2048 loss, parameters, throughput, and cost.

## Acceptance Criteria

- Results use same-stack measurements and distinguish MoE total/active parameters.
- Unsupported shapes fail explicitly; this task does not launch final training.
