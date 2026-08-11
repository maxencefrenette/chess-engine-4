---
id: "01kzsacs6q"
title: "Calibrate AdamH learning rates for the dense 16d batch"
status: completed
priority: high
effort: medium
dependencies: []
tags: ["training", "dense", "optimizer", "hyperball"]
created_at: 2026-08-11
completed_at: 2026-08-11
---

# Calibrate AdamH learning rates for the dense 16d batch

## Objective

Restore an explicitly calibrated AdamH learning-rate recipe when adaptive dense
training selects batch `16d`. Test whether a compact law predicts the selected
rates across widths; retain a width-indexed table when it does not.

## Tasks

- [x] Inventory the existing matched `16d` AdamH runs and retain them as sweep
  cells rather than duplicating paid work.
- [x] Fill the missing BF16 learning-rate cells at d64, d128, d256, and d512.
- [x] Find spike-free MXFP8 rates at d768 and d1024 before extrapolating to
  d1280.
- [x] Test any proposed law on held-out widths and reject it if it changes the
  selected discrete LR cell or materially worsens final EMA task loss.
- [x] Validate d1280 at exact batch `16d` after the lower-width relationship is
  known.
- [x] Record every run, W&B URL, cost, loss, spike count, `EG_flops`, and the
  promotion verdict in `experiments/2026-08-11.04-adamh-16d-learning-rates`.
- [x] Implement the user-requested explicit `16d` table separately from the
  retained `32d` table.
- [x] Rerun the affected canonical 0.055x d256 and d512 cells with the final
  interpolated 16d learning rates, compare them, and promote them as explicit
  user-designated current-recipe overrides.
- [x] Rerun and promote the affected d256/d512 0.1x off-arm cells so every
  scaling-fit observation that selects 16d uses the active LR table.

## Evidence

- Twenty-one new exact-`16d`, `0.055x`, seed-1 runs cost approximately `$7.110`
  from recorded W&B runtimes and repository hardware rates.
- The final conservative table is d64 `0.005`, d128 `0.0035`, d256 `0.0022`,
  d512 `0.0013`, d768 `0.001`, d1024 `0.00044`, and d1280 `0.00044`.
- D256 and d512 are user-requested, rounded log-log interpolations between the
  retained d128 `0.0035` and d768 `0.001` anchors; they are intentionally not
  represented as measured optima.
- Their canonical 0.055x reruns were both spike-free. D256 reached loss
  `3.388976` and EG_flops `0.911x`; d512 reached loss `3.184742` and EG_flops
  `0.721x`. Both failed the ordinary higher-EG promotion rule, but the user
  explicitly designated them canonical so the registry reflects the active
  conservative recipe rather than the superseded higher-LR observations.
- The reruns cost approximately `$0.130`, bringing new experiment work to about
  `$7.240`.
- The affected 0.1x off-arm reruns were also spike-free: d256 loss `3.289253`,
  EG_flops `0.965x`; d512 loss `3.091250`, EG_flops `0.767x`. They replace the
  superseded higher-LR rows by the same explicit current-recipe override.
- The off-arm refresh cost approximately `$0.209`, bringing new experiment work
  to about `$7.449`.
- A d768/d1024 power fit predicted approximately `0.00023` at held-out d1280.
  The stable `0.00022` run was `0.03844` loss worse than stable `0.00031`, so
  the proposed law failed the `0.005` gate. Stable `0.00044` improved further.
- Every selected LR is spike-free and is bracketed on the tested grid; the next
  higher MXFP8 cells each recorded one spike.
- Report: `experiments/2026-08-11.04-adamh-16d-learning-rates`.
- Verification: `uv run pytest -q` (226 passed), `uv run ruff check .`,
  `pnpm --dir website lint`, `pnpm --dir website build`, `taskmd validate`, and
  `git diff --check` all pass.

## Acceptance Criteria

- All comparisons use batch `16d`, ratio `0.055x`, seed 1, identical accepted
  samples, losses, schedule, data, and optimizer implementation; only LR varies.
- A selected LR is spike-free, per the repository optimizer-selection rule.
- A law is promoted only if it predicts a held-out width's selected LR on the
  tested `sqrt(2)` grid and remains competitive within `0.005` final EMA task
  loss; otherwise use an explicit table.
- Paid stages print and review a conservative cost estimate before launch.
- Focused config tests cover both the `16d` and `32d` LR selections.
- `uv run pytest -q`, `uv run ruff check .`, website lint/build,
  `taskmd validate`, and `git diff --check` pass before completion.
