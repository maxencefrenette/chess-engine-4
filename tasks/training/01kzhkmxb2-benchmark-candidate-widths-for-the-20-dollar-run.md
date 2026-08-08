---
id: "01kzhkmxb2"
title: "Benchmark candidate widths for the 20 dollar run"
status: pending
priority: high
effort: medium
dependencies: []
tags: ["training", "scaling", "benchmark"]
created_at: 2026-08-08
---

# Benchmark candidate widths for the 20 dollar run

## Objective

Measure only the missing model shapes that could change the final `$20`
selection. Dense d1536 is already a supported recipe width; MoE d1536 remains
a feasibility candidate and must not become canonical without evidence.

## Tasks

- [ ] Refresh or confirm the retained d1024 and d2048 throughput baselines on
  the current training stack.
- [ ] Add dense d1536 to the measured throughput sweep and validate its
  production batch, precision, memory use, and step time on B200.
- [ ] Test whether MoE d1536 fits and trains stably on B200; add recipe support
  only if the model and batch satisfy the existing routing and kernel contracts.
- [ ] Compare d1024, d1536, and d2048 candidates by predicted loss, total and
  active parameters, samples required, runtime, and realized cost.
- [ ] Run a short quality calibration only if the updated value-of-information
  calculation says it can materially change the final selection.
- [ ] Record the benchmark and selection consequences in an experiment report.

## Acceptance Criteria

- Every candidate considered by the planner has same-stack measured throughput.
- The report distinguishes `N_total` from `N_active` for MoE candidates.
- Unsupported or memory-unsafe shapes are rejected explicitly rather than
  silently omitted.
- The result identifies the viable frontier; it does not launch the final run.
