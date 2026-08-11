---
id: "01kzn0mj58"
title: "Evaluate quantile load balancing for MoE routing"
status: completed
priority: medium
effort: medium
dependencies: []
tags: ["training", "moe", "routing", "experiment"]
created_at: 2026-08-09
---

# Evaluate quantile load balancing for MoE routing

## Objective

Test quantile load balancing against the canonical top-2 MoE router and its
auxiliary load-balancing loss.

## Tasks

- [x] Define and test the exact quantile-routing algorithm.
- [x] Train matched baseline and quantile-balanced candidates.
- [x] Compare loss, EG_flops, expert utilization, stability, and throughput.
- [x] Write an experiment report.
- [x] After review, preserve the QB bias through export/inference and cut over the MoE family.

## Acceptance Criteria

- Model, accepted samples, training FLOPs, and task-loss weights are matched;
  the load-balancing term differs by design.
- Per-layer routing distributions and dead experts are reported.
- Automatic promotion improves EG_flops without a policy or stability
  regression; the final QB cutover is an explicit reviewed override because the
  isolated spikes recovered immediately and the method removes a tuned loss weight.

## Progress

Implementation and retained evidence are in
`experiments/2026-08-10.03-quantile-load-balancing/`. QB improved EG_flops at
d256 and d512 but not d128. Fresh matched auxiliary-loss controls reproduced
that result at all three widths. After review, QB became the sole MoE routing
strategy, the d256 and d512 winners were promoted, and the learned QB bias was
added to Safetensors export and the lc0 ce4 inference router. The production
SM120 build and an exported d256 backend evaluation both passed.
