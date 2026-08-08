---
id: "01kzg0rdwf"
title: "Evaluate position subsampling for value diversity"
status: in-progress
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

## Progress

The raw-archive capacity blocker was superseded by the user's corrected design. A deterministic
row-identity sampler now streams canonical Parquet directly without derived datasets or game-boundary
assumptions. A 497-shard startup manifest fixes the exact input snapshot despite concurrent atomic
corpus appends. The three authorized 15,000-step runs were stopped after live single-thread loader
throughput invalidated the sub-$1.50 cost estimate; none completed or produced a final checkpoint.
The retained report proposes an eight-thread, matched 8,500-step replacement plan, but replacement
runs require explicit user authorization.

## Source

[Lc0 Net](https://app.notion.com/p/35a8054d223580b79ebadc55321dd4d3)
