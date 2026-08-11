---
id: "01kzs3w5xb"
title: "Validate Muon at d768 and retune d256 AdamW"
status: completed
priority: medium
effort: medium
dependencies: []
tags: ["optimizer", "experiment", "validation"]
created_at: 2026-08-11
completed_at: 2026-08-11
---

# Validate Muon at d768 and retune d256 AdamW

## Objective

Test whether the d256 realized-efficiency result survives a matched AdamW LR
retune, and measure batched Muon convergence and runtime at d768.

## Tasks

- [x] Benchmark the batched Muon optimizer at d768 on B200.
- [x] Retune AdamW near the canonical d256 LR at the matched `0.055x` horizon.
- [x] Run lower-LR batched Muon arms at d768.
- [x] Recompute optimizer and realized-efficiency verdicts from spike-free runs.

## Acceptance Criteria

- AdamW and Muon comparisons keep model, data, seed, batch, loss, and horizon fixed.
- Learning-rate selection excludes runs with loss spikes.
- The report records W&B URLs, `EG_flops`, runtime, cost, and whether the prior d256 claim survives.
