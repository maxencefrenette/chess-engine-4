---
id: "01m4p21n3g"
title: "Refresh website dependencies and repository maintenance"
status: completed
priority: medium
effort: medium
dependencies: []
tags: ["maintenance", "website", "training", "docs"]
created_at: 2026-10-11
---

# Refresh website dependencies and repository maintenance

## Objective

Refresh website dependencies, share the dense training schedule between recipes
and planning, and repair development/data documentation while another project
has priority on the workstation CPU. Leave uv on `latest` and do not add CI.

## Tasks

- [x] Update Next.js, its ESLint config, React, and the website lockfile.
- [x] Share dense sample-budget and minimum-step scheduling rules.
- [x] Document native CPU-test setup and lightweight/full verification commands.
- [x] Record completed Iceberg ingestion and remaining reader/upload work; remove
      the obsolete SM90 blocker from the toolchain task.

## Acceptance Criteria

- Existing dense schedules and learning-rate choices remain unchanged.
- Website dependencies install from the lockfile and pass lint/build.
- Focused scheduling tests and the repository verification gate pass with local
  CPU affinity and low process priority. Do not launch training or benchmarks.
- Completed ingestion evidence is preserved in `experiments/`; historical
  reports and the existing Modal corpus metadata are not rewritten.

## Verification — 2026-10-11

- Next.js / eslint-config-next 16.4.0, React / React DOM 19.3.0, and
  Tailwind / its PostCSS plugin 4.3.3. Refreshed vulnerable transitive lockfile
  entries within supported dependency ranges; frozen offline install passes.
- Focused recipe/planner tests: 109 passed. Full Python suite: 236 passed.
- `mise run check-fast`: Ruff, ty, and diff whitespace checks passed.
- Website lint and production build passed, including TypeScript and all 11
  generated static pages. Build used one worker; checks used one CPU core at
  nice level 19 with numerical-library thread caps. No training, benchmarks,
  native rebuilds, CI, or uv pinning changes.
- Production pnpm audit: no known vulnerabilities. Full audit retains one
  unpatched development-only `braces` advisory, recorded for upstream follow-up
  in `01kzj8kfvz`; no forced dependency override.
- Dashboard watcher includes the new shared scheduling helper. Generated
  dashboard data has no tracked changes.
