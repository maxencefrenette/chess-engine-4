---
id: "01kzmbxvts"
title: "Evaluate mixing deeply evaluated openings into training"
status: pending
priority: medium
effort: medium
dependencies: []
tags: ["training", "data", "openings", "experiment"]
created_at: 2026-08-09
---

# Evaluate mixing deeply evaluated openings into training

## Objective

Test whether mixing broad, high-node opening evaluations into LCZero self-play
data improves opening play and overall Elo.

## Tasks

- [ ] Generate and deduplicate all positions through three plies, every UHO
      start, and other explicitly defined opening sets.
- [ ] Evaluate them with a frozen engine, network, node count, and settings.
- [ ] Compare mixture rates at matched rows and compute, then run Elo tests.
- [ ] Write an experiment report.

## Acceptance Criteria

- The dataset manifest records coverage, provenance, and evaluator settings.
- The report separates opening metrics from overall Elo.
- No mixture is promoted without a statistically credible Elo gain.
