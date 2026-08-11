---
id: "01kzp3qnrb"
title: "Evaluate the Skaling law for chess training"
status: completed
priority: medium
effort: medium
dependencies: []
tags: ["research", "scaling-laws", "training"]
created_at: 2026-08-10
completed_at: 2026-08-10
---

# Evaluate the Skaling law for chess training

## Objective

Determine whether the coupled scaling law and sparse profiling strategy from
[Skaling](https://arxiv.org/abs/2608.07222) improve this project's predictions.

## Tasks

- [x] Fit Skaling and current baselines to canonical dense and MoE runs.
- [x] Compare held-out interpolation and model/data extrapolation error.
- [x] Test total versus active parameter count explicitly for MoE.
- [x] Expand dense coverage and compare full-grid and anchored fits.
- [x] Integrate the supported dense fit into budget planning.

## Acceptance Criteria

- Fits use identical runs, loss targets, and cross-validation splits.
- Prediction error and uncertainty are reported by extrapolation regime.
- Recommend adopt, reject, or collect a small set of missing runs.

## Outcome

- Report: `experiments/2026-08-10.01-skaling-law`.
- Thirty-five profiling runs used about `$2.826` of recorded GPU time.
- Dense Skaling improved full-fit MAPE from `0.797%` to `0.254%` and
  interpolation MAPE from `0.610%` to `0.256%`.
- Dense planning uses the curated d64-d1024 surface; MoE, `compare-run`, and the
  website retain their prior laws.
- D64 remains the fit floor.
