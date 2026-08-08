---
id: "01kzg0rdd6"
title: "Tighten tournament Elo confidence intervals"
status: completed
priority: high
effort: medium
dependencies: []
tags: ["evaluation", "notion-import"]
touches: ["evaluation", "experiments"]
created_at: 2026-08-07
completed_at: 2026-08-08
---

# Tighten tournament Elo confidence intervals

## Objective

Reduce the uncertainty of policy-Elo and low-visit tournament results without
wasting games on badly mismatched engines. Determine whether paired-game
pentanomial statistics and the Swiss scheduler can produce materially tighter
confidence intervals for the same GPU cost.

## Tasks

- [x] Audit the tournament output retained for paired-game information.
- [x] Implement statistically appropriate rating and confidence-interval reporting.
- [x] Compare the new estimator with the current estimator on retained tournaments.
- [x] Validate against the retained 100-pair searched PGN; no new tournament was needed.

## Acceptance Criteria

- Reports Elo and a clearly defined confidence interval for each engine.
- Uses paired-game information when the tournament design provides it.
- Demonstrates the uncertainty change at matched game count or matched cost.

## Source

[Lc0 Net](https://app.notion.com/p/35a8054d223580b79ebadc55321dd4d3)
