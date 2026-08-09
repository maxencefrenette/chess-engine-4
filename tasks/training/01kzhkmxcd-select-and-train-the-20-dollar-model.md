---
id: "01kzhkmxcd"
title: "Select and train the 20 dollar model"
status: pending
priority: critical
effort: large
dependencies: ["01kzg0rdwf", "01kzhkmxbr"]
tags: ["training", "final-run", "budget"]
created_at: 2026-08-08
---

# Select and train the 20 dollar model

## Objective

Select and train the strongest supported model whose planned steady-state GPU
and CPU training cost is `$20`, using the verified corpus and current measured
throughput. Preliminary experiments are accounted for separately from the final
run, and realized Modal cost is reported separately from the planner estimate.

## Tasks

- [ ] Rerun the budget planner with the verified usable sample count and fresh
  candidate throughput measurements.
- [ ] Compare the leading candidates, uncertainty intervals, sample limits,
  stability risks, and inference implications.
- [ ] Freeze the exact config, steps, sample count, seed, data manifest, commit,
  and cost basis in a final-run plan.
- [ ] Review the printed Modal launch summary with the user before allowing the
  expensive run to proceed.
- [ ] Train once, retaining periodic and final checkpoints and complete W&B
  metrics.
- [ ] Check task loss, loss spikes, gradients, dead experts, samples, runtime,
  and realized cost before declaring the run valid.

## Acceptance Criteria

- The user approves the exact launch summary and budget basis before training.
- Planned steady-state GPU and CPU cost is at most `$20`; startup and realized
  billing differences are reported explicitly.
- The final checkpoint is complete and recoverable from the Modal artifact
  volume.
- The run has no disqualifying instability under the current promotion rules.
