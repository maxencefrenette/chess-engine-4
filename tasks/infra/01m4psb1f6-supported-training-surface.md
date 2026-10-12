---
id: "01m4psb1f6"
title: "Retire unsupported training options and cloud budget planners"
status: completed
priority: medium
effort: large
dependencies: []
tags: ["maintenance", "training"]
created_at: 2026-10-11
---

# Retire unsupported training options and cloud budget planners

## Objective

Retire NVFP4 training, non-SwiGLU production activation choices, and cloud-dollar
budget/ladder planners. Separate deployment choices from canonical recipes and
document historical scripts against a verified historical revision.

## Tasks

- [x] Remove retired precision/activation options from production training.
- [x] Move cloud GPU/CPU selection and loader defaults out of canonical recipes.
- [x] Remove budget/ladder commands, implementation, and command-specific tests.
- [x] Document revision-bound historical analysis instead of restoring retired APIs.
- [x] Verify recipe parity, supported launch paths, and the full repository gate.

## Acceptance Criteria

- Model/optimizer/batch/step/loss defaults at every canonical width are unchanged.
- Local deployment detects the GPU; Modal deployment retains its existing choices.
- Historical evidence, portable analysis of old activations, and inference support
  remain intact. New production training rejects unsupported options early.
- Checkpoint configuration loading remains compatible with persisted `infra` fields.
- No training, benchmarks, native rebuilds, or cloud jobs during verification;
  use one low-priority CPU core and bounded numerical-library threads.
- Ruff, ty, CPU suites with/without Modal, website lint/build, task and diff checks pass.

## Verification — 2026-10-11

- All 14 canonical configurations, resolved with Modal defaults, matched the
  pre-change serialized baseline exactly. Tests cover explicit deployment
  overrides, local selection, old checkpoint fields, and retired-option rejection.
- Modal-extra suite: 250 passed. With Modal imports blocked in a fresh
  interpreter: 237 passed, 4 cloud-specific modules skipped.
- Ruff, ty, uv lock check, website lint/build, task validation, and diff checks
  passed. Website build regenerated identical dashboard JSON (SHA-256
  `838091ff57d972223b32a99f00c0de3b7815a1eb92a2499d06b9e1c413ccc5b9`).
- Verified the historical script paths, inputs, and fitting definitions at the
  full revision recorded in `experiments/historical-scripts.toml`; did not rerun
  historical plots or change reports/results.
- Checks used one low-priority CPU core with numerical threads capped. No
  training, benchmarks, native rebuilds, or cloud jobs were launched.
