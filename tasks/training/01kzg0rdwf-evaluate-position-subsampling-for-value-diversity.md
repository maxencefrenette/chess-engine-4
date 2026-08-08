---
id: "01kzg0rdwf"
title: "Evaluate position subsampling for value diversity"
status: pending
priority: high
effort: medium
dependencies: []
tags: ["value", "data", "experiment", "notion-import"]
touches: ["training", "data", "experiments"]
created_at: 2026-08-07
---

# Evaluate position subsampling for value diversity

## Objective

Test whether sampling roughly 1-10% of positions from each game increases game
and value-target diversity enough to improve value learning at matched training
FLOPs or matched dollar cost.

## Tasks

- [ ] Define deterministic sampling that avoids accidental game-length bias.
- [ ] Measure the resulting game, position, and value-target distributions.
- [ ] Train matched candidates at several sampling rates.
- [ ] Compare task loss components, EG_flops, and realized cost.

## Acceptance Criteria

- The report distinguishes diversity gains from a change in sample count.
- The chosen sampling rule is deterministic and reproducible.
- Any promoted rule improves the stated objective without a policy regression.

## Source

[Lc0 Net](https://app.notion.com/p/35a8054d223580b79ebadc55321dd4d3)
