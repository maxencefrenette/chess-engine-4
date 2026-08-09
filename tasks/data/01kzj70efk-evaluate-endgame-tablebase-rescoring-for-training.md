---
id: "01kzj70efk"
title: "Evaluate endgame tablebase rescoring for training targets"
status: blocked
priority: low
effort: large
dependencies: []
tags: ["data", "training", "tablebase", "desktop"]
created_at: 2026-08-08
---

# Evaluate endgame tablebase rescoring for training targets

## Objective

Determine whether the project should reproduce LCZero's endgame tablebase
rescoring of training positions, and implement a compatible offline pipeline if
the evidence supports it. This work must run on the user's desktop PC, where the
required tablebases and local compute will be available, rather than on Modal.

## Tasks

- [ ] Inspect the current upstream LCZero training-data rescoring implementation
      and document exactly which positions and targets it changes.
- [ ] Inventory the desktop PC's tablebase coverage, storage, CPU throughput, and
      source-data access after the machine is brought back online.
- [ ] Design an incremental and resumable rescoring pass with immutable source
      provenance and explicit output validation.
- [ ] Compare rescored and original targets on a representative corpus slice.
- [ ] If the target changes are material, train and evaluate matched candidates
      before promoting rescored data into the canonical corpus.

## Acceptance Criteria

- The implementation is grounded in the actual LCZero rescoring behavior rather
  than an inferred approximation.
- Original training data remains recoverable and every changed target is
  attributable to a tablebase probe.
- The pipeline can resume safely after interruption and reports coverage,
  throughput, failures, and target-change counts.
- Adoption requires downstream training and playing-strength evidence, not only
  successful preprocessing.

## Blocker

Blocked until the user brings the desktop PC back online. The task depends on
that machine's local tablebase storage and compute capacity.
