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

Expand and validate the canonical Parquet corpus for final training.

## Tasks

- [x] Inventory existing data and upstream source overlap.
- [x] Convert and upload shards with resumable, exact-set, auditable commands.
- [x] Verify shard integrity, targets, provenance, and production-loader compatibility.
- [x] Update canonical corpus metadata only after validation.

## Acceptance Criteria

- [x] Every added shard passed exact source/output verification before cleanup.
- [x] The production loader reads the final corpus with valid targets.
- [x] Corpus metadata and live source manifests are complete.

## Result

- Expanded from `480` to `1,203` shards and from `3,949,735,220` to
  `8,020,779,820` rows.
- Final Parquet size is `696,169,217,477` bytes; combined live Volume usage was
  `745,014,201,497` bytes.
- All `723` added sources have unique names and SHA-256 manifests under Modal
  `/source-manifests`; no tar, temporary, or partial files remain.
- The all-shard target audit found no invalid or non-finite values.
- No training run was launched.
