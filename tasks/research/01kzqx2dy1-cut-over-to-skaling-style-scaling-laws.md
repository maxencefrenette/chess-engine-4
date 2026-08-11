---
id: "01kzqx2dy1"
title: "Cut over to Skaling-style scaling laws"
status: completed
priority: high
effort: medium
dependencies: []
tags: ["research", "scaling-laws", "training", "website"]
created_at: 2026-08-11
completed_at: 2026-08-11
---

# Cut over to Skaling-style scaling laws

## Objective

Use the coupled `L(N,D)` Skaling law for every active validation-loss prediction
path, with total parameters as `N` and the dense loss floor reused by MoE.

## Plan

- [x] Promote the quantile-balanced MoE surface to canonical scaling evidence.
- [x] Add measured d384 and d640 throughput to the canonical MoE ladder.
- [x] Remove the legacy FLOPs-only and undertraining loss laws.
- [x] Use Skaling for dense and MoE budget planning, uncertainty, and VOI.
- [x] Use Skaling for run comparison and website loss curves.
- [x] Regenerate generated website data and verify the complete repository.

## Acceptance criteria

- Dense and MoE both fit `SkalingLaw` from canonical `scaling_runs`.
- MoE shares dense `E`; all other coefficients remain family-specific.
- Budget planning uses total parameters and samples directly for every family.
- Run comparison computes same-width sample-equivalent `EG_flops` from Skaling.
- No active production path imports the retired loss laws.
- Website loss curves are generated from `(N,D)` predictions.
- Full repository tests, Ruff, website lint/build, task validation, and diff check pass.

## Outcome

- Dense and MoE both use `SkalingLaw`; MoE shares dense `E = 2.3994`.
- The QB MoE surface has 11 canonical points spanning d256-d1024 and
  `0.01x-0.25x`; d128 and auxiliary-router evidence remain excluded.
- D384 and d640 have canonical measured B200 throughput rows.
- Budget planning, bootstrap uncertainty, VOI, run comparison, and website loss
  curves use `(N,D)` directly.
- The old FLOPs-only loss and undertraining-law implementations were removed.
- Current central budget winners are dense d768 at `$1`, MoE d512 at `$5`, and
  MoE d768 at `$10`; bootstrap selection uncertainty remains material.
