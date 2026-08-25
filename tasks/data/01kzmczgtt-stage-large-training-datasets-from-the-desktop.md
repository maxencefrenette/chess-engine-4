---
id: "01kzmczgtt"
title: "Stage large training datasets from the desktop"
status: in-progress
priority: low
effort: medium
dependencies: []
tags: ["data", "desktop", "modal", "storage"]
created_at: 2026-08-09
---

# Stage large training datasets from the desktop

## Objective

Prepare larger LCZero datasets on the desktop, upload them to Modal only for
large training runs.

## Tasks

- [ ] Convert each LCZero tar into exactly one corresponding Parquet shard.
- [ ] Build resumable upload and verification commands for large runs.

## Acceptance Criteria

- Interrupted transfers resume without corruption or duplicate shards.
- Every uploaded Parquet shard maps directly to one source tar.
- Training starts only after every uploaded shard verifies successfully.

## Progress

- 2026-08-24: Desktop storage is available at `/data/chess` with 14.4 TiB free.
- Froze the complete upstream t80 listing at 9,958 archives and
  8,992,666,368,000 advertised bytes.
- Started the resumable serial source download under `/data/chess/t80/source`
  with an advertised-size check, atomic completion, retry backoff, a 64 MiB/s
  rate ceiling, and 512 GiB free-space reserve.
- Migrated the desktop downloader from shell to `scripts/download_t80.py` while
  preserving its frozen inventory and on-disk resume state.
- Parquet conversion and Modal upload remain deferred until their methodology is
  revisited.
