---
id: "01kzmczgtt"
title: "Stage large training datasets from the desktop"
status: blocked
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

## Blocker

Wait for the desktop PC to be set up again.
