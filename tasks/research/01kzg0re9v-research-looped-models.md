---
id: "01kzg0re9v"
title: "Research looped models"
status: pending
priority: low
effort: medium
dependencies: []
tags: ["speculative", "architecture", "notion-import"]
created_at: 2026-08-07
---

# Research looped models

## Objective

Reassess looped models if evidence emerges that training with M recurrent passes
and inferring with N greater than M passes improves chess strength more
efficiently than allocating the same inference compute to MCTS.

## Tasks

- [ ] Review evidence for train-time and inference-time loop-count mismatch.
- [ ] Compare recurrent-depth compute with additional MCTS nodes and memory use.
- [ ] Identify stability, halting, and LCZero backend implications.
- [ ] Recommend reject, defer, or run a bounded experiment.

## Acceptance Criteria

- The recommendation compares against MCTS as the actual inference-compute baseline.
- No implementation proceeds without a plausible compute-efficiency advantage.

## Source

[Lc0 Net](https://app.notion.com/p/35a8054d223580b79ebadc55321dd4d3)
