---
id: "01kzqr5dm7"
title: "Calibrate adaptive dense batch size and minimum steps"
status: completed
priority: high
effort: medium
dependencies: []
tags: ["training", "dense", "optimization", "scaling-laws"]
created_at: 2026-08-10
---

# Calibrate adaptive dense batch size and minimum steps

## Objective

Calibrate the learning-rate adjustment for reducing the dense batch from `32d`
to `16d`, then establish a conservative minimum optimizer-step rule. The dense
recipe should choose the largest validated batch and reject a requested horizon
when neither batch can provide enough optimizer steps.

## Tasks

- [x] Tune and transfer-check the `16d` learning-rate adjustment against the
  retained `32d` learning-rate law.
- [x] Measure matched `16d` and `32d` horizons across model scales, preserving
  accepted samples within every pair.
- [x] Estimate conservative minimum-step boundaries and bracket them across
  scales.
- [x] Obtain user review of the proposed fitted threshold.
- [x] Implement automatic batch selection and reject horizons below the
  validated minimum.
- [x] Confirm the d1280 multiplier at the exact `16d` batch. The exact-batch
  `1.30x` run recorded two spikes, which the user explicitly reviewed and
  accepted for promotion.
- [x] Record commands, W&B URLs, costs, metrics, uncertainty, and the promotion
  verdict in `experiments/2026-08-10.04-dense-minimum-steps`.

## Evidence

- Forty Modal runs cost `$7.404` from recorded runtimes and repository rates.
- The promoted `16d / 32d` LR dictionary uses d128/d512 `0.85x`, d768 `1.00x`,
  d1024 `1.15x`, and d1280 `1.30x`; the lower d512-d1024 values removed the
  short-horizon spikes.
- Direct minimum-step boundaries were measured at d128, d512, d768, and d1024.
- The reviewed rule uses the smooth power law fitted from the `32d` crossings
  for both the batch-selection and rejection cutoffs.
- Report: `experiments/2026-08-10.04-dense-minimum-steps`.
- Verification: `uv run pytest -q` (202 passed), `uv run ruff check .`,
  `pnpm --dir website lint`, `pnpm --dir website build`, and
  `git diff --check` all pass.
- D1280 provisionally selected `1.30x` over `1.15x`: final EMA loss
  `2.8680` versus `2.8768`, with zero versus one loss spike. The two-run
  extension cost `$2.218`, but used batch 24,576 rather than exact `16d`.
- The exact d1280 `16d` minimum-horizon confirmation used batch 20,480 for
  20,238 steps. It reached EMA loss `2.8946` and `EG_flops 1.340x`; the user
  explicitly accepted its two spikes and promoted the run. It cost `$0.837`.

## Acceptance Criteria

- Total estimated hardware spend is reviewed before each stage, targets less
  than `$5`, and never exceeds the authorized `$8` ceiling.
- Learning-rate comparisons keep model, accepted samples, data, seed, schedule,
  and loss configuration matched.
- `32d` is selected only when its optimizer-step count clears the validated
  conservative boundary; otherwise `16d` is selected.
- Configurations are rejected when `16d` cannot clear the validated minimum.
- Focused tests cover the two batch choices, exact sample preservation, and the
  rejection boundary.
- No canonical recipe change is promoted before user review of the experimental
  evidence.
