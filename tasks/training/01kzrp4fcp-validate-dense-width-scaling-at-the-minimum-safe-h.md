---
id: "01kzrp4fcp"
title: "Validate dense width scaling at the minimum safe horizon"
status: completed
priority: high
effort: medium
dependencies: []
tags: ["training", "dense", "scaling-laws"]
created_at: 2026-08-11
---

# Validate dense width scaling at the minimum safe horizon

## Objective

Determine whether the adaptive dense batch recipe can recover the model-width
component of the dense scaling law with a profiling horizon approximately half
the previous `0.1x` width arm.

## Tasks

- [x] Preflight the exact `0.05x` ladder through the production rejection rule.
- [x] Run the closest common safe horizon with matched ratio and recipe.
- [x] Compare the short-horizon width fit with the established `0.1x` arm and
  its held-out predictions.
- [x] Record cost, W&B URLs, stability, fit error, and verdict.
- [x] Retry the three spiked widths at the next lower LR multiplier, promote
  stable values into the recipe, and recompute the fit.

## Acceptance Criteria

- Do not override a minimum-step rejection.
- Keep training ratio, seed, loss, schedule, and data recipe fixed across widths.
- Compare width exponent identifiability and held-out prediction error, not only
  training completion.
- Do not promote short-horizon observations before user review.

## Outcome

- Exact `0.05x` is rejected at d256-d1024; `0.055x` is the closest clean common
  ratio and uses automatic `16d` selection at all six widths.
- Lowering the LR multipliers to d512 `0.85x`, d768 `1.00x`, and d1024 `1.15x`
  removed all three spikes; those values are promoted into the recipe.
- The stable short-arm fit improves held-out MAPE from `1.221%` to `0.725%`.
- Nine runs cost `$1.579`; no canonical scaling points were promoted.
- Report: `experiments/2026-08-11.01-dense-half-horizon-width-scaling`.
- Verification: 202 tests, Ruff, website lint/build, task validation, and
  `git diff --check` pass.
