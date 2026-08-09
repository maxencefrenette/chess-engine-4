---
id: "01kzhkmxbr"
title: "Expand and validate the final training corpus"
status: completed
priority: high
effort: large
dependencies: ["01kzg0rdwf"]
tags: ["data", "modal", "training"]
created_at: 2026-08-08
completed_at: 2026-08-09
---

# Expand and validate the final training corpus

## Objective

Build enough verified, sufficiently diverse training rows for the selected
`$20` candidate without exceeding Modal storage or treating duplicated epochs
as new data. The required row count comes from the final candidate, not from a
fixed aspirational corpus size.

## Tasks

- [x] Measure current Parquet bytes per position and remaining Modal-volume
  capacity before downloading or converting more data.
- [x] Inventory available source data, canonical source-name overlap, anomalous
  files, and archive-level duplicate risk.
- [x] Coordinate the deterministic position-subsampling rule with dependency
  `01kzg0rdwf` without implementing or launching that experiment here.
- [x] Convert and upload shards with resumable, exact-set, auditable commands.
- [x] Verify schema, row counts, target distributions, shard integrity,
  production-loader compatibility, and archive-level accidental duplication.
- [x] Update `experiments/training-data.toml` only after the usable corpus is
  confirmed.

## Acceptance Criteria

- [x] The corpus contains at least the unique rows required by the approved
  final launch plus operational headroom.
- [x] Exact positions, shards, bytes per position, and storage headroom are
  recorded. Exact total games are explicitly unavailable because the user
  required verified transient-source deletion and Parquet omits game identity.
- [x] The production loader reads every final shard and retains its canonical
  measured-throughput configuration; no training run was launched.
- [x] Source data was retained through exact conversion verification, then
  deleted only after verification per the user's explicit override of the
  original source-retention criterion.

## Progress

- Modal's official pricing includes `1 TiB/month` across workspace storage.
  The operational ceiling remained `900 GiB` (`966,367,641,600` bytes)
  combined, preserving `124 GiB` below the paid boundary.
- The pre-expansion canonical corpus was `480` shards, `3,949,735,220` rows,
  and `342,720,758,136` Parquet bytes. Training plus artifacts used
  `690,294,698,209` bytes.
- The upstream inventory exposed `9,181` nontrivial archives. The anomalous
  `training-run1-test80-20240428-1817.tar` was `1,454,080` bytes and remained
  excluded by the `100 MiB` minimum.
- Expansion used `723` unique archives totaling `836,396,574,720` bytes. Every
  source matched advertised bytes and SHA-256; provenance is recorded in
  `experiments/2026-08-08.01-training-corpus-capacity/sources.toml` and live
  `/source-manifests` JSON.
- Serial acquisition committed exact sync-run manifests before downloading,
  reserved a source-sized conversion allocation, remeasured both Volumes, and
  rejected outside-run complete sources. A stopped Modal app was safely
  reattached to its original exact selection via `--sync-run-id`.
- Every new Parquet shard was atomically renamed, never overwritten, compared
  against its source through the production native loader, and reverified
  before source cleanup.
- Final audit: `1,203` shards, `8,020,779,820` rows, `696,169,217,477` Parquet
  bytes, and `86.795702` bytes/row. This exceeds the quarter-rate experiment's
  `5,223,219,200` raw-position requirement by `2,797,560,620` positions.
- Final Volume usage: training `696,169,464,367` bytes, artifacts
  `48,844,737,130` bytes, combined `745,014,201,497` bytes (`693.849 GiB`).
  Headroom is `221,353,440,103` bytes (`206.151 GiB`); no tar, tmp, or Parquet
  partial remains.
- A production-loader audit sampled `307,968` rows across every shard and found
  zero non-finite targets, invalid Q/D ranges, invalid implied WDL triples, or
  negative moves-left values.
- Provenance reconciliation found `723/723` unique source names, `723/723`
  unique SHA-256 values, and `723/723` expected canonical outputs. The first
  `17` sources had `1,143,453` games and zero duplicate game IDs. Corpus-wide
  game-level duplication cannot be reconstructed after user-directed source
  cleanup because Parquet intentionally omits game IDs; this limitation is
  recorded rather than guessed away.
