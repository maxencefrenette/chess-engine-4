---
id: "01kzg0rdta"
title: "Evaluate truncated top-k policy targets"
status: pending
priority: medium
effort: large
dependencies: []
tags: ["policy", "data", "experiment", "notion-import"]
touches: ["training", "data", "evaluation", "experiments"]
created_at: 2026-08-07
---

# Evaluate truncated top-k policy targets

## Objective

Determine whether retaining only the top 16 or top 32 MCTS policy targets can
reduce dataset size without materially reducing playing strength. Keep the
softmax denominator over every legal move.

## Tasks

- [ ] Add an experimental top-k policy transformation without changing stored data.
- [ ] Train matched full-policy, top-16, and top-32 candidates.
- [ ] Compare held-out policy loss and top-1 accuracy.
- [ ] Compare candidates in a low-visit tournament with enough search to test tails.
- [ ] Measure the storage reduction available if truncation is adopted on disk.

## Acceptance Criteria

- Makes a retain-or-reject decision using downstream playing strength.
- Does not change the Parquet schema unless the experiment succeeds.
- Reports quality, throughput, and projected storage effects.

## Source

[Lc0 Net](https://app.notion.com/p/35a8054d223580b79ebadc55321dd4d3)
