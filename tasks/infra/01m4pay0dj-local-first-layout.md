---
id: "01m4pay0dj"
title: "Separate local model workflow from cloud and research tooling"
status: completed
priority: medium
effort: large
dependencies: []
tags: ["maintenance", "training", "publishing"]
created_at: 2026-10-11
---

# Separate local model workflow from cloud and research tooling

## Objective

Make local training and custom-lc0 model export independent of Modal; separate
cloud adapters and research analysis from the production training package.
Preserve CLI command names, model numerics, historical results, and engine support.

## Tasks

- [x] Move cloud adapters and make the Modal dependency optional.
- [x] Relocate analysis tooling and rename portable reference models.
- [x] Separate local model export from explicit remote export.
- [x] Rewrite the entrypoint documentation around the local custom-lc0 workflow.
- [x] Verify local imports without Modal and run the repository gate at low priority.

## Acceptance Criteria

- Local training, export, evaluation helpers, and offline analysis import without
  the Modal SDK or construction of cloud images/volumes.
- Missing local checkpoints fail locally; remote export requires an explicit flag.
- Existing command names remain available; research scripts follow renamed imports.
- No training, benchmarks, native rebuilds, or cloud jobs during verification.
- Tests, Ruff, ty, website lint/build, task validation, and diff checks pass.
- Iceberg integration, resume, local evaluation execution, and full release
  packaging remain separate follow-up work; do not imply they are implemented.

## Progress and verification — 2026-10-11

- Moved cloud adapters into `backends/modal/`, research tooling into `analysis/`,
  and model export into `release/export.py`. Renamed the portable model module
  to `model/reference.py`. The SDK is now an optional `modal` extra; cloud image
  dependency installation includes it explicitly.
- Extracted cloud tournament/benchmark runners from reusable evaluation code;
  match payloads receive the engine path explicitly. Existing opening-book
  defaults and result identities are unchanged.
- Preserved all 29 CLI command names. Local export no longer falls back to Modal
  for missing files; remote export requires `--remote`, and `--remote-only`
  requires that flag. Other module paths and repository imports were updated.
- Historical reports, registries, recipes, and numerical behavior are unchanged.
  Four historical analysis scripts received import-path-only edits.
- Full Modal-extra CPU suite: 242 passed; the subsequently added cloud-entrypoint
  wiring test also passed (three focused wiring/local-boundary checks passed).
  Full suite with the Modal SDK deliberately blocked: 230 passed, four
  cloud-specific test modules skipped. No cloud jobs were submitted.
- Ruff, ty, lockfile check, task validation, document-link checks, and whitespace
  checks passed. Website lint and production build passed with one page worker.
- Tests/builds ran at nice level 19 on one CPU core with numerical-library thread
  caps. No training, benchmarks, native rebuilds, or Docker changes.
- README now describes the local-to-custom-lc0 workflow. Detailed cloud,
  research, data, and development instructions live in `docs/`; no CI was added,
  and uv remains on `latest`.
