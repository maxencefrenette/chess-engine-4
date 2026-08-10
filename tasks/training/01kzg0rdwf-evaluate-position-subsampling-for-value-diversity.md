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

- [x] Define fresh random, scheduling-independent row sampling.
- [x] Audit exact retained and batch-usable row capacity.
- [x] Train matched candidates at retention 1.0, 0.5, and 0.25.
- [x] Compare task loss components, EG_flops, stability, runtime, and cost.
- [x] Run a complete 800-visit mirrored-opening round robin with paired Elo intervals.

## Acceptance Criteria

- The report distinguishes diversity gains from a change in sample count.
- The chosen sampling rule draws a fresh random subset for each launch.
- Any promoted rule improves the stated objective without a policy regression.

## Progress

Fresh-random replication completed on the same 497-shard corpus state. All three
arms consumed 983,040,000 accepted rows in 15,000 steps; EMA task loss ranked
0.25 (2.835924), 0.5 (2.847041), then 1.0 (2.865800). A 4,416-game / 2,208-pair
random-UHO tournament at 800 visits reproduced the searched ranking: quarter
beats half by +32.17 Elo [19.68, 44.66], and half beats full by +42.74
[31.47, 54.02]. Completed-game tournament cost was $1.804934. The quarter run
had two automatic spike flags, so random subsampling is retained for review but
0.25 is not promoted pending a clean checkpoint.

Reopened after the user clarified that each treatment should use a fresh random
row sample rather than fixed row-identity membership. The original matched
1.0/0.5/0.25 training and searched UHO tournament will be repeated to test
whether the ranking replicates.

Reopened after the user requested migration of all Elo tournaments to randomly selected,
mirrored `UHO_Lichess_4852_v1` openings and approximately $2 of additional 800-node games.
The evaluator now uses a pinned, hashed 65,536-position UHO sample, seeded shuffle, explicit
opening offsets/identities, and native ce4 cross-game batches averaging 176.8-241.7 positions.
The completed 5,552-game / 2,776-pair extension plus probe cost $1.9775. Its 1,304-cluster fit
ranks quarter retention +25.35 Elo over half (95% CI [13.66, 37.05]) and half +35.58 over full
([23.89, 47.28]). Quarter retention is recommended for review without canonical promotion.

The raw-archive capacity blocker was superseded by the user's corrected design. A deterministic
row-identity sampler streams canonical Parquet without derived datasets or game-boundary assumptions.
Three standard eight-thread runs completed at exactly
15,000 steps / 983,040,000 accepted samples each. The user authorized the full quarter run after
clarifying that $1.50 was not strict. Half and quarter retention improved loss and searched play
over full retention. A complete 384-game, 192-pair round robin at 800 visits ranked half first and
quarter second with overlapping intervals. The retained experiment report recommends half retention
for review but makes no canonical promotion.

## Source

[Lc0 Net](https://app.notion.com/p/35a8054d223580b79ebadc55321dd4d3)
