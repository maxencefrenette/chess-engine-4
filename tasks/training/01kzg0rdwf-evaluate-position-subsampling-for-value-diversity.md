---
id: "01kzg0rdwf"
title: "Evaluate position subsampling for value diversity"
status: completed
priority: high
effort: medium
dependencies: []
tags: ["value", "data", "experiment", "notion-import"]
touches: ["training", "data", "experiments"]
created_at: 2026-08-07
completed_at: 2026-08-09
---

# Evaluate position subsampling for value diversity

## Objective

Test whether fresh random position subsampling increases value-target diversity
enough to improve value learning and searched play at matched accepted samples
and training FLOPs.

## Tasks

- [x] Define fresh random Bernoulli row sampling.
- [x] Train matched candidates at sampling rates 1.0, 0.5, and 0.25.
- [x] Compare task loss components, EG_flops, stability, runtime, and cost.
- [x] Run a complete 800-visit mirrored-opening round robin with paired Elo intervals.

## Acceptance Criteria

- The report distinguishes diversity gains from a change in sample count.
- The chosen sampling rule draws a fresh random subset for each launch.
- Any promoted rule improves the stated objective without a policy regression.

## Source

[Lc0 Net](https://app.notion.com/p/35a8054d223580b79ebadc55321dd4d3)
