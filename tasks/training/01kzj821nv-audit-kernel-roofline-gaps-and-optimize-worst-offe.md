---
id: "01kzj821nv"
title: "Audit kernel roofline gaps and optimize worst offenders"
status: blocked
priority: high
effort: large
dependencies: ["01kzhp5w47", "01kzj8kfvz"]
tags: ["cuda", "kernels", "roofline", "optimization", "benchmark"]
created_at: 2026-08-08
---

# Audit kernel roofline gaps and optimize worst offenders

## Objective

Measure retained kernels against their attainable rooflines, then optimize the
worst high-impact gaps.

## Tasks

- [ ] Benchmark and rank all retained SM80/90/100/120 training and inference
      kernels by normalized roofline gap and end-to-end importance.
- [ ] Optimize the worst worthwhile gaps without changing supported semantics.
- [ ] Recheck neighboring shapes and canonical training or LCZero inference.

## Acceptance Criteria

- The audit covers the complete retained matrix with reproducible measurements.
- Promote only numerically correct changes that improve the relevant end-to-end path.

## Blocker

Wait for SM90 support and the toolchain refresh.
