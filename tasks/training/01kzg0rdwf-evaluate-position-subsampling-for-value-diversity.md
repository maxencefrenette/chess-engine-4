---
id: "01kzg0rdwf"
title: "Evaluate position subsampling for value diversity"
status: blocked
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

## Blocker

The strongest planner-selected configuration strictly below the `$1.50` steady-state GPU+CPU
ceiling needs 1,305,804,800 unique rows per treatment. The quarter-rate arm therefore needs about
5.2 billion raw positions, while the three isolated matched datasets alone exceed the remaining
900 GiB operational headroom. No run was launched. See
`experiments/2026-08-08.01-position-subsampling/README.md` for the audited arithmetic and resume
conditions.

The final corpus audit at commit `c678bcb` retained 17 hashed sources with 125,250,708 positions
and no duplicate game IDs. Keep this task stopped until the user either authorizes a sequential
streaming design that may delete verified temporary raw archives, or shrinks the experiment to a
storage-limited model and matched row target.

## Source

[Lc0 Net](https://app.notion.com/p/35a8054d223580b79ebadc55321dd4d3)
