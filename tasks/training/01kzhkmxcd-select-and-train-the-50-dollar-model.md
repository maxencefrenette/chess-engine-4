---
id: "01kzhkmxcd"
title: "Select and train the 50 dollar model"
status: pending
priority: critical
effort: large
dependencies: ["01kzg0rdwf", "01kzhkmxbr", "01kzj70efk"]
tags: ["training", "final-run", "budget"]
created_at: 2026-08-08
---

# Select and train the 50 dollar model

## Objective

Select and train the strongest supported dense model at a roughly `$50`
training-cost target, using the new tablebase-rescored dataset and current
measured throughput. MoE is deferred.

## Tasks

- [ ] Rerun dense-only budget planning with the new dataset and fresh throughput.
- [ ] Freeze the config, seed, sample count, dataset manifest, and cost basis.
- [ ] Review the printed Modal launch summary with the user before allowing the
      expensive run to proceed.
- [ ] Train once, retain checkpoints and metrics, and validate run stability.

## Acceptance Criteria

- The user approves the exact launch summary and budget basis before training.
- The final run uses the audited tablebase-rescored dataset.
- Planned GPU and CPU cost targets roughly `$50`; realized cost is reported.
- The checkpoint is recoverable and has no disqualifying instability.
