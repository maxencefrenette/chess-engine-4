---
id: "01kzg0re0r"
title: "Improve Modal experiment queueing"
status: pending
priority: medium
effort: large
dependencies: []
tags: ["modal", "automation", "notion-import"]
touches: ["training"]
created_at: 2026-08-07
---

# Improve Modal experiment queueing

## Objective

Replace ad hoc parallel Modal launch and retry handling with a small, durable
queueing workflow that respects concurrency and budget constraints while
keeping each run independently inspectable.

## Tasks

- [ ] Document failure modes in the current sweep launch workflow.
- [ ] Design queue state that survives local process interruption.
- [ ] Enforce configured concurrency and avoid duplicate launches.
- [ ] Provide clear retry and cancellation behavior.

## Acceptance Criteria

- A sweep can be launched once and safely resumed after interruption.
- Concurrency limits are enforced without polling or duplicate jobs.
- Failed, running, completed, and cancelled runs remain auditable.

## Source

[Lc0 Net](https://app.notion.com/p/35a8054d223580b79ebadc55321dd4d3)
