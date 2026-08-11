---
id: "01kzg0re2y"
title: "Automate canonical results promotion"
status: pending
priority: medium
effort: large
dependencies: []
tags: ["automation", "experiments", "notion-import"]
touches: ["training", "experiments", "website"]
created_at: 2026-08-07
---

# Automate canonical results promotion

## Objective

Automate the mechanical parts of evaluating a W&B run and updating canonical
best-run TOML data, while retaining an explicit human decision for promotions
that change methodology or optimize a nonstandard objective.

## Tasks

- [ ] Produce a structured promotion result from the existing comparison logic.
- [ ] For learning-rate tuning, exclude runs with loss spikes and require the
      selected learning rate to be spike-free.
- [ ] For other parameter studies, record loss spikes without rejecting the
      parameter result and require follow-up learning-rate tuning for the
      selected configuration.
- [ ] Update the correct family and width entry atomically after approval.
- [ ] Validate the resulting TOML and generated website data.

## Acceptance Criteria

- Routine promotions no longer require agents to edit best-runs TOML manually.
- The update records the run URL, metrics, recipe identity, and promotion rationale.
- Historical experiment reports remain immutable.

## Source

[Lc0 Net](https://app.notion.com/p/35a8054d223580b79ebadc55321dd4d3)
