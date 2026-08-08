---
id: "01kzhp5w47"
title: "Add SM90 training and inference support"
status: pending
priority: high
effort: large
dependencies: []
tags: ["cuda", "sm90", "modal", "training", "inference"]
created_at: 2026-08-08
---

# Add SM90 training and inference support

## Objective

Add H100 and H200 as supported SM90 targets for the existing training and lc0
inference workflows, without changing canonical GPU selections before cost
benchmarks exist.

## Tasks

- [ ] Add H100 and H200 to training hardware validation and Modal launch paths.
- [ ] Add an SM90 inference build and runtime path using the simplest correct
      retained implementation.
- [ ] Preserve explicit `--gpu` experiment overrides and reject unsupported
      precision, backend, and architecture combinations clearly.
- [ ] Add focused correctness and configuration tests for both GPU identifiers.

## Acceptance Criteria

- Dense and MoE training can be launched explicitly on supported H100 and H200
  configurations.
- An exported supported network runs through the lc0 inference backend on SM90.
- No model recipe selects H100 or H200 until the separate cost benchmark task
  establishes that it is cheapest.
- Focused tests cover device identity, configuration round trips, dispatch, and
  unsupported combinations.
