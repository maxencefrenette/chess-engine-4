---
id: "01kzn0mj58"
title: "Evaluate quantile load balancing for MoE routing"
status: pending
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

- [ ] Define and test the exact quantile-routing algorithm.
- [ ] Train matched baseline and quantile-balanced candidates.
- [ ] Compare loss, EG_flops, expert utilization, stability, and throughput.
- [ ] Write an experiment report and Elo-test any promotable candidate.

## Acceptance Criteria

- Model, accepted samples, training FLOPs, and loss weights are matched.
- Per-layer routing distributions and dead experts are reported.
- Any promotion improves EG_flops without a policy or stability regression.
