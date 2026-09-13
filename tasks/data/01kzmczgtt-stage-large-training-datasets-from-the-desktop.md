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

- [ ] Ingest each LCZero tar as one resumable unit into the partitioned Iceberg table.
- [ ] Build resumable upload and verification commands for large runs.

## Acceptance Criteria

- Interrupted transfers resume without corruption or duplicate shards.
- Every Iceberg row records its source tar, and each source tar is committed atomically.
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
- 2026-09-06: Resumed the desktop download after an interrupted run at 2,084 of
  9,958 archives (2,558,588,221,440 bytes complete).
- 2026-09-08: Resumed after a host reboot at 3,384 of 9,958 archives; verified
  archive 3,385 completed and the downloader continued to archive 3,386.
- 2026-09-09: Resumed at 3,398 of 9,958 archives in a detached session; verified
  archive 3,399 completed and the downloader continued to archive 3,400.
- Parquet conversion and Modal upload remain deferred until their methodology is
  revisited.
- 2026-09-12: Selected an Iceberg schema with normalized packed inputs, compact
  policy data, root/best/outcome targets, piece count, and source-archive
  provenance. The table uses `truncate[4](num_pieces)` partitioning; one source
  archive may therefore produce files in several partitions.
- 2026-09-13: Ingested the first 10 archives into the local `training.t80`
  Iceberg table. The test contains 82,490,581 positions in 90 data files across
  nine piece-count partitions and occupies 7,983,095,732 data bytes. Conversion
  sustained about 93,000 positions/s per worker and produced 96.78 compressed
  bytes/position, or 47.1% of source-tar size. On a matched archive, the previous
  schema used 86.77 bytes/position, so the selected columns and partitioning add
  11.5%. An interrupted run resumed by skipping six already committed archives;
  the completed table has no leftover partial files.
