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
---

# Evaluate position subsampling for value diversity

## Objective

Test whether deterministic position subsampling increases value-target diversity
enough to improve value learning and searched play at matched accepted samples
and training FLOPs.

## Tasks

- [x] Define deterministic, scheduling-independent row sampling.
- [x] Audit exact retained and batch-usable row capacity.
- [x] Train matched candidates at retention 1.0, 0.5, and 0.25.
- [x] Compare task loss components, EG_flops, stability, runtime, and cost.
- [x] Run a complete 800-visit mirrored-opening round robin with paired Elo intervals.

## Acceptance Criteria

- The report distinguishes diversity gains from a change in sample count.
- The chosen sampling rule is deterministic and reproducible.
- Any promoted rule improves the stated objective without a policy regression.

## Progress

Reopened after the user requested migration of all Elo tournaments to randomly selected,
mirrored `UHO_Lichess_4852_v1` openings and approximately $2 of additional 800-node games.
The evaluator now uses a pinned, hashed 65,536-position UHO sample, seeded shuffle, explicit
opening offsets/identities, and native ce4 cross-game batches averaging 176.8-241.7 positions.
The completed 5,552-game / 2,776-pair extension plus probe cost $1.9775. Its 1,304-cluster fit
ranks quarter retention +25.35 Elo over half (95% CI [13.66, 37.05]) and half +35.58 over full
([23.89, 47.28]). Quarter retention is recommended for review without canonical promotion.

The raw-archive capacity blocker was superseded by the user's corrected design. A deterministic
row-identity sampler streams canonical Parquet without derived datasets or game-boundary assumptions,
and a 497-shard manifest fixes provenance. Three standard eight-thread runs completed at exactly
15,000 steps / 983,040,000 accepted samples each. The user authorized the full quarter run after
clarifying that $1.50 was not strict. Half and quarter retention improved loss and searched play
over full retention. A complete 384-game, 192-pair round robin at 800 visits ranked half first and
quarter second with overlapping intervals. The retained experiment report recommends half retention
for review but makes no canonical promotion.

## Source

[Lc0 Net](https://app.notion.com/p/35a8054d223580b79ebadc55321dd4d3)
