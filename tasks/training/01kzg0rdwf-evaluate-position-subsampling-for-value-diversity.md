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
An authorized 500-step quarter-rate probe initially showed 1.59-1.60M accepted samples/s, so the
user authorized standard eight-thread replacement runs at matched 15,000 steps / 983,040,000
accepted samples. Full and half completed validly for $1.2448 and $1.2214 and were exported. The
quarter arm's longer standalone window sustained only ~1.0-1.17M samples/s, projecting $1.8-$1.9;
it was stopped under the strict $1.50 cap without a final checkpoint. Completion and the required
three-candidate 800-node tournament are blocked pending authorization of a hard $2.00 quarter cap.

## Source

[Lc0 Net](https://app.notion.com/p/35a8054d223580b79ebadc55321dd4d3)
