---
id: "01kzhkmxbr"
title: "Expand and validate the final training corpus"
status: pending
priority: high
effort: large
dependencies: ["01kzg0rdwf"]
tags: ["data", "modal", "training"]
created_at: 2026-08-08
---

# Expand and validate the final training corpus

## Objective

Build enough verified, sufficiently diverse training rows for the selected
`$20` candidate without exceeding Modal storage or treating duplicated epochs
as new data. The required row count comes from the final candidate, not from a
fixed aspirational corpus size.

## Tasks

- [ ] Measure current Parquet bytes per position and remaining Modal-volume
  capacity before downloading or converting more data.
- [ ] Inventory available source games and positions and identify overlap with
  the current 3,949,735,220-position corpus.
- [ ] Apply the accepted deterministic position-subsampling rule during source
  conversion, preserving game-diversity accounting.
- [ ] Convert and upload shards with resumable, auditable commands.
- [ ] Verify schema, row counts, target distributions, shard integrity, loader
  throughput, and absence of accidental duplicates.
- [ ] Update `experiments/training-data.toml` only after the usable corpus is
  confirmed.

## Acceptance Criteria

- The corpus contains at least the unique rows required by the approved final
  launch plus operational headroom.
- Exact positions, games, shards, bytes per position, and storage headroom are
  recorded.
- The production loader sustains the selected model's measured throughput.
- Source data is retained until conversion and training validation complete.
