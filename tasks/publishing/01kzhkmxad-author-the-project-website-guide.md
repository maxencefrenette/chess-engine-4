---
id: "01kzhkmxad"
title: "Author the project website guide"
status: pending
priority: high
effort: large
parent: "01kzg0rdfm"
dependencies: ["01kzhkmx9s"]
tags: ["website", "documentation", "experiments"]
created_at: 2026-08-08
---

# Author the project website guide

## Objective

Implement the approved website brief as a source-driven guide to how Chess
Engine 4 works, while preserving the existing experiment explorer and static
build.

## Tasks

- [ ] Explain the LCZero data contract and Rust/Parquet input pipeline.
- [ ] Explain dense as the current path and MoE as a deferred experiment,
  including the evidence that motivated the decision.
- [ ] Explain Modal training, precision, custom kernels, checkpoints, export,
  lc0 inference, evaluation, and canonical-run promotion.
- [ ] Integrate retained experiment evidence and scaling charts without
  rewriting historical reports.
- [ ] Add repository and experiment links so technical claims are traceable.
- [ ] Test the content and navigation at desktop and mobile sizes.

## Acceptance Criteria

- An external technical reader can follow data to training to export to
  evaluation without repository knowledge.
- Quantitative claims are generated from or linked to canonical evidence.
- The static production build contains no secrets or private artifact paths.
- The approved narrative and page hierarchy are implemented faithfully.
