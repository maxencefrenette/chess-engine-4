---
id: "01kzqgc085"
title: "Complete the MoE Skaling low-compute edge"
status: completed
priority: medium
effort: medium
dependencies: []
tags: ["research", "scaling-laws", "training", "moe"]
created_at: 2026-08-10
completed_at: 2026-08-10
---

# Complete the MoE Skaling low-compute edge

## Objective

Determine whether a matched low-compute model-size edge makes the Skaling law
reliable enough for MoE64A2 budget planning, using less than `$5` of new
experiments.

## Plan

- [x] Train d128, d768, and d1024 at `0.01x` with the canonical MoE64A2
  recipe, roughly step-matching the dense `0.05x` floor.
- [x] Preserve W&B URLs, runtime, loss, spikes, and router health.
- [x] Refit Chinchilla and Skaling on identical total- and active-parameter data.
- [x] Compare interpolation, size extrapolation, data extrapolation, and
  L-shaped holdout error.
- [x] Adopt MoE Skaling only if extrapolation and parameter stability support it.

## Acceptance Criteria

- New recorded hardware cost is below `$5` using measured GPU and CPU rates.
- The three new runs differ only in width and derived canonical settings.
- The matched `0.01x` size edge covers d128, d256, d512, d768, and d1024.
- The report distinguishes total from active parameters and records whether any
  fitted exponent hits its bound.
- Canonical MoE budget planning is unchanged unless held-out evidence improves.

## Progress

- A d768 `0.02x` launch was stopped when the floor design changed. Modal app
  `ap-awVyZhWpwjvgLM4T5u5RcM` ran for about 11.5 minutes; it is not fit data,
  but its cost counts toward the experiment ceiling.

## Outcome

- The completed follow-up plus the interrupted launch recorded `$3.845` of
  GPU-plus-CPU runtime.
- D128 `0.01x` ended with seven dead experts and is excluded from fitting.
- Total-parameter Skaling improved full-fit and interpolation MAPE, but lost
  size extrapolation (`0.943%` versus `0.784%`) and kept `alpha` at its bound.
- The `0.01x` sparse edge had `2.819%` held-out MAPE, only marginally better
  than Chinchilla's `2.851%`.
- Canonical MoE allocation remains unchanged.
