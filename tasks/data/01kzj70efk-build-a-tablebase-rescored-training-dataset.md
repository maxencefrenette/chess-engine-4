---
id: "01kzj70efk"
title: "Build a tablebase-rescored training dataset"
status: blocked
priority: critical
effort: large
dependencies: []
tags: ["data", "training", "tablebase", "desktop"]
created_at: 2026-08-08
---

# Build a tablebase-rescored training dataset

## Objective

Reproduce LCZero's tablebase rescoring on the desktop PC using the public
[`noobpwnftw/chesstb`](https://huggingface.co/buckets/noobpwnftw/chesstb)
bucket, then generate a new immutable full dataset before the final run.

## Tasks

- [ ] Freeze the bucket inventory and checksums, then match LCZero's rescoring behavior.
- [ ] Build a resumable desktop pipeline that never mutates the source dataset.
- [ ] Generate and audit a complete new Parquet dataset and manifest.
- [ ] Train matched original/rescored candidates and run the approved Elo tournament.
- [ ] Write the experiment report and update final-run data only if Elo improves.

## Acceptance Criteria

- Every changed target is attributable to a probe from the frozen bucket manifest.
- The new dataset is complete, reproducible, independently manifested, and audited.
- The matched experiment and report demonstrate an Elo gain before final training.
- The original dataset remains untouched and recoverable.

## Blocker

Wait for the desktop PC and its tablebases to be available.
