---
id: "01kzg0re7r"
title: "Research memory layers"
status: pending
priority: low
effort: large
dependencies: []
tags: ["speculative", "architecture", "notion-import"]
created_at: 2026-08-07
---

# Research memory layers

## Objective

Assess whether memory layers such as UltraMemV2 can improve chess-network
training or inference efficiency within this project's dense and sparse model
families.

## Tasks

- [ ] Review the paper and any reproducible implementation.
- [ ] Map its parameter, active-compute, and memory behavior to this project.
- [ ] Evaluate compatibility with LCZero inference and custom kernels.
- [ ] Recommend reject, defer, or run a bounded experiment.

## Acceptance Criteria

- States a concrete mechanism by which memory layers could help this workload.
- Includes total and active parameter implications and implementation cost.

## Source

[Lc0 Net](https://app.notion.com/p/35a8054d223580b79ebadc55321dd4d3) and
[UltraMemV2](https://arxiv.org/abs/2508.18756)
