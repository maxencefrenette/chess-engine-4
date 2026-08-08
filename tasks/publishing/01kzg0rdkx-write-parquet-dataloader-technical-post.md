---
id: "01kzg0rdkx"
title: "Write parquet dataloader technical post"
status: pending
priority: low
effort: medium
dependencies: []
tags: ["writing", "data", "notion-import"]
created_at: 2026-08-07
---

# Write parquet dataloader technical post

## Objective

Document the migration from LCZero tar records to the Rust and Parquet data
pipeline, including format choices, correctness checks, storage savings, and
measured training-throughput effects.

## Tasks

- [ ] Summarize the old data path and its bottlenecks.
- [ ] Explain the Parquet schema and Rust conversion/loading design.
- [ ] Present storage, loader-throughput, and end-to-end training measurements.
- [ ] Publish the post on the project website.

## Acceptance Criteria

- Measurements are traceable to retained experiment reports.
- The post covers correctness validation as well as performance.

## Source

[Lc0 Net](https://app.notion.com/p/35a8054d223580b79ebadc55321dd4d3)
