---
id: "01kzg0rdyk"
title: "Design a compact lower-precision training format"
status: pending
priority: low
effort: large
dependencies: []
tags: ["speculative", "data", "storage", "notion-import"]
touches: ["data", "training"]
created_at: 2026-08-07
---

# Design a compact lower-precision training format

## Objective

When storage becomes the limiting constraint, replace the current Parquet
representation with a compact format targeting roughly 80 bytes per position
and 8 billion positions below the 1 TiB Modal storage limit. This task is
explicitly deferred until that trigger is reached.

## Tasks

- [ ] Validate reversible board-history reconstruction from board state and move deltas.
- [ ] Compare multinomial policy sampling at k=1, 4, and 8 with full soft targets.
- [ ] Try Gumbel top-k only if multinomial sampling leaves meaningful quality on the table.
- [ ] Quantize root Q, D, and moves-left targets within explicit error bounds.
- [ ] Prototype chunked sharding, compression, legal-move regeneration, and Rust loading.
- [ ] Measure bytes per position, loader throughput, and matched training quality.

## Acceptance Criteria

- Stores 8 billion positions below 800 GB with operational headroom.
- Preserves the LCZero input and legal-policy loss contracts.
- Meets the current loader's end-to-end training throughput or justifies any regression.
- Includes a migration and validation plan before deleting source data.

## Source

[Lc0 Net](https://app.notion.com/p/35a8054d223580b79ebadc55321dd4d3) and
[Compact Training Data Format](https://app.notion.com/p/39a8054d223581e2ae36f5dfd8dfcfbf)
