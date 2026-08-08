---
id: "01kzhkmxbr"
title: "Expand and validate the final training corpus"
status: blocked
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

- [x] Measure current Parquet bytes per position and remaining Modal-volume
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

## Progress

- Modal's current Volume allowance is `1 TiB/month` across the workspace's
  volumes. This is `1,099,511,627,776` bytes, not one decimal terabyte.
- The pre-expansion live audit found `342,720,758,136` bytes in the training
  volume and `347,573,940,073` bytes in the artifact volume, or
  `690,294,698,209` bytes combined.
- Footer metadata independently confirms the canonical corpus has
  `3,949,735,220` rows in `480` Parquet shards, using `86.770565` bytes per
  row.
- The operational ceiling is `900 GiB` (`966,367,641,600` bytes) combined,
  retaining `124 GiB` below the paid boundary. Every acquisition preflight
  includes both volumes and reserves one source-sized output allocation for
  every retained but unconverted tar.
- The official test80 listing exposed `9,181` nontrivial archives when
  inventoried. Canonical source-name overlap accounts for `432` of the existing
  shards; the other `48` older canonical sources are no longer listed upstream.
  The anomalous `training-run1-test80-20240428-1817.tar` is only `1,454,080`
  bytes and is excluded.
- Canonical conversion remains gated on the accepted result of dependency
  `01kzg0rdwf`. Sources are retained at the volume root with exact URL, byte
  size, and SHA-256 provenance under `/source-manifests`; experimental prefixes
  and canonical `/parquet` remain untouched.
- Acquisition stopped with `17` retained sources totaling `25,723,576,320`
  bytes. Final combined usage is `716,018,279,187` bytes, leaving
  `250,349,362,413` bytes below the 900 GiB operational ceiling.
- Full decoding confirms the retained sources contain `1,143,453` games and
  `125,250,708` v6 positions. All tar/gzip members and record boundaries are
  valid, and no canonical game ID repeats across the 17 archives.
- Dependency `01kzg0rdwf` requires `1,305,804,800` quarter-rate rows, or about
  `5,223,219,200` raw positions. The reserve policy admits at most
  `138,036,471,695` source bytes; at the audited `205.356241` compressed bytes
  per raw position this is about `168,045,139` quarter-rate rows, `7.7706x`
  short. No safe canonical conversion target is available without changing the
  experiment or storage constraint.
