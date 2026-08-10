---
id: "01kzp3qnrb"
title: "Evaluate the Skaling law for chess training"
status: pending
priority: medium
effort: medium
dependencies: []
tags: ["research", "scaling-laws", "training"]
created_at: 2026-08-10
---

# Evaluate the Skaling law for chess training

## Objective

Determine whether the coupled scaling law and sparse profiling strategy from
[Skaling](https://arxiv.org/abs/2608.07222) improve this project's predictions.

## Tasks

- [ ] Fit Skaling and current baselines to canonical dense and MoE runs.
- [ ] Compare held-out interpolation and model/data extrapolation error.
- [ ] Test total versus active parameter count explicitly for MoE.
- [ ] Assess whether an L-shaped run grid would improve budget planning.

## Acceptance Criteria

- Fits use identical runs, loss targets, and cross-validation splits.
- Prediction error and uncertainty are reported by extrapolation regime.
- Recommend adopt, reject, or collect a small set of missing runs.
