---
id: "01kzrr2xyj"
title: "Add L-shaped scaling ladder planner and 16d throughput"
status: pending
priority: high
effort: medium
dependencies: []
tags: ["scaling", "budget", "throughput"]
created_at: 2026-08-11
---

# Add L-shaped scaling ladder planner and 16d throughput

## Objective

Make scaling-law acquisition follow a canonical L-shaped ladder before spending
budget on optional interior points, and give the planner measured throughput for
the adaptive dense recipe's half-batch variant.

## Tasks

- [x] Add a canonical dense ladder grid and a `plan-ladder` command.
- [x] Count every recorded coordinate toward scaffold coverage, including spiked runs.
- [x] Restrict optional value-of-information suggestions to unobserved grid cells.
- [x] Benchmark and cache half-batch dense throughput at all supported widths.
- [ ] Replace the stale d1280 `19.2d` profile with an exact `16d` profile.
- [x] Make planner costs and launch commands resolve the batch selected by the recipe.
- [x] Add focused tests and run the repository verification gate.

## Acceptance Criteria

- The command reports missing d64 data-arm and 0.055x width-arm cells before
  considering optional runs.
- Once complete, the command ranks only cheap unobserved cells inside the grid.
- Spiked observations satisfy scaffold coverage without being silently promoted
  as stable recipe evidence.
- Dense half-batch costs use measured end-to-end throughput.
- Planner commands preserve the recipe's samples, batch choice, and step guardrail.
- Focused and broad verification pass.
