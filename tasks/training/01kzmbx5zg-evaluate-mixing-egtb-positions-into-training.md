---
id: "01kzmbx5zg"
title: "Evaluate mixing EGTB positions into training"
status: pending
priority: medium
effort: medium
dependencies: ["01kzj70efk"]
tags: ["training", "data", "tablebase", "experiment"]
created_at: 2026-08-09
---

# Evaluate mixing EGTB positions into training

## Objective

Test whether adding exact tablebase positions from common material imbalances
to LCZero self-play data improves playing strength.

## Tasks

- [ ] Build a reproducible, deduplicated EGTB sample covering common imbalances.
- [ ] Compare several mixture rates at matched rows and training compute.
- [ ] Run the approved Elo tournament and write an experiment report.

## Acceptance Criteria

- Coverage, sampling weights, exact targets, and provenance are recorded.
- The report separates overall Elo from endgame-specific performance.
- No mixture is promoted without a statistically credible Elo gain.

## Dependency

Reuse the desktop tablebase pipeline from `01kzj70efk`.
