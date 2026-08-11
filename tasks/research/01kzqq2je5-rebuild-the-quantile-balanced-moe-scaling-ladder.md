---
id: "01kzqq2je5"
title: "Rebuild the quantile-balanced MoE scaling ladder"
status: completed
priority: high
effort: medium
dependencies: []
tags: ["research", "scaling-laws", "training", "moe"]
created_at: 2026-08-10
completed_at: 2026-08-11
---

# Rebuild the quantile-balanced MoE scaling ladder

## Objective

Collect a clean low-compute scaling ladder for the canonical quantile-balanced
MoE64A2 recipe without mixing observations from the retired auxiliary-loss
router.

## Plan

- [x] Add canonical recipe support for d384 and d640 on B200 with TE MXFP8.
- [x] Price `0.01x` and `0.02x` model-size bands before launching paid work.
- [x] Select `0.01x` as the model-size profiling ratio after cost review.
- [x] Profile the new d384 and d640 widths before training them.
- [x] Train d256, d384, d512, d640, d768, and d1024 at `0.01x`.
- [x] Train the d256 horizon band at `0.01x`, `0.02x`, `0.05x`, `0.1x`, and
  `0.25x`, counting the shared model-band corner only once.
- [x] Preserve W&B URLs, runtime, loss, spikes, and router health for every run.
- [x] Fit and validate the MoE scaling law using only quantile-balanced runs.

## Cost plan

The current no-launch estimate is approximately `$5.6` for a `0.01x`
model-size band plus the d256 horizon band, or `$9.0` for a `0.02x` model-size
band plus the same horizon band. These estimates scale measured quantile-balanced
d256/d512 runtimes, legacy-router d768/d1024 runtime only as hardware timing
evidence, and interpolated B200 step times for the unprofiled d384/d640 widths.
Authorize at least `$6.5` or `$10.5`, respectively, to cover profiling and timing
uncertainty. No paid runs are authorized by this plan.

## Acceptance criteria

- Every fit observation uses the canonical quantile-balanced router.
- D128 is excluded from the fitting ladder.
- The model band contains all six widths from d256 through d1024.
- The d256 horizon band spans `0.01x` through `0.25x` with round ratios.
- Training launches are preceded by fresh d384/d640 profiles and reviewed
  launch summaries.
- Actual GPU-plus-CPU cost and any interrupted work are recorded.
- Model-size and training-horizon extrapolation are evaluated separately.

## Outcome

- The two profiles plus nine new training runs cost `$5.712` in measured
  GPU-plus-CPU runtime, below the `$6.50` reviewed allowance.
- All runs completed with zero dead experts. D1024 `0.01x` and d256 `0.25x`
  each recorded one recovered spike; the other seven recorded none.
- D256 `0.25x` training-horizon extrapolation is accurate to `0.183%` for both
  Chinchilla and Skaling when withheld from the boundary fit.
- Chinchilla beats Skaling on d1024 model-size extrapolation, `0.326%` versus
  `0.989%` MAPE.
- Reusing dense `E` improves full MAPE to `0.386%`, but both MoE exponents still
  hit their upper bounds. The model-size law remains unidentified.
- Both laws miss the off-sweep canonical d512 `0.05x` point by about `1.8%`.
  No new run is promoted and MoE remains excluded from allocation planning.
