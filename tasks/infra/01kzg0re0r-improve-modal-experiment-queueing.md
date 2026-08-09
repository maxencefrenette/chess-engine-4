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
keeping each run independently inspectable. In particular, submitting work
while all 10 workspace GPUs are occupied should leave the work waiting without
holding a paid GPU, rather than failing the experiment launch.

## Tasks

- [ ] Document failure modes in the current sweep launch workflow.
- [ ] Determine which Modal-native queueing, concurrency-limit, and retry
      primitives can wait durably for workspace GPU capacity.
- [ ] Design queue state that survives local process interruption.
- [ ] Enforce configured concurrency and avoid duplicate launches.
- [ ] Distinguish queued-for-capacity jobs from started, failed, and retryable
      jobs without treating quota exhaustion as an experiment failure.
- [ ] Provide clear retry and cancellation behavior.

## Acceptance Criteria

- A sweep can be launched once and safely resumed after interruption.
- When all 10 GPUs are in use, additional jobs wait and later start without
  manual relaunch, duplicate execution, or consuming a GPU while queued.
- Concurrency limits are enforced without fragile local polling or duplicate
  jobs.
- Waiting jobs retain their exact config, launch summary, budget, and input
  manifest, and can be inspected or cancelled.
- Failed, running, completed, and cancelled runs remain auditable.

## Source

[Lc0 Net](https://app.notion.com/p/35a8054d223580b79ebadc55321dd4d3)
