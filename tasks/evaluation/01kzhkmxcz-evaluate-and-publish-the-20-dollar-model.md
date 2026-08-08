---
id: "01kzhkmxcz"
title: "Evaluate and publish the 20 dollar model"
status: pending
priority: high
effort: medium
dependencies: ["01kzg0rdd6", "01kzhkmxcd"]
tags: ["evaluation", "experiments", "website"]
created_at: 2026-08-08
---

# Evaluate and publish the 20 dollar model

## Objective

Validate the completed `$20` checkpoint end to end, measure its playing strength
and inference cost, and publish the result without promoting an invalid run.

## Tasks

- [ ] Export the checkpoint to the stable Safetensors format.
- [ ] Perform tensor-level or tightly bounded output comparisons between the
  training model and lc0 backend.
- [ ] Benchmark supported inference GPUs and realistic batch sizes.
- [ ] Run the approved tournament protocol with the tightened confidence-
  interval estimator.
- [ ] Compare loss, policy accuracy, policy/search Elo, nodes per second, and
  training/inference cost with the incumbent ladders.
- [ ] Write the experiment report and promote canonical registries only if the
  applicable criteria are met.
- [ ] Add the validated result to the public website data and narrative.

## Acceptance Criteria

- Export and lc0 inference agree within documented numerical tolerances.
- Elo is reported with a clearly defined confidence interval and retained raw
  match evidence.
- The report states whether the run improved training efficiency, playing
  strength, and realized cost separately.
- Website and canonical registry changes reflect only validated evidence.
