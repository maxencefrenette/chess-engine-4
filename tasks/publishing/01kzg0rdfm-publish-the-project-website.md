---
id: "01kzg0rdfm"
title: "Publish the project website"
status: pending
priority: high
effort: medium
dependencies: []
tags: ["website", "notion-import"]
touches: ["website"]
created_at: 2026-08-07
---

# Publish the project website

## Objective

Publish the existing Next.js experiment explorer as the public presentation of
the project. Keep generated scaling data automatic and make the deployed site
useful without exposing secrets or requiring a Python server.

This is a high-priority project direction, but its implementation plan still
needs collaborative iteration and decomposition before work begins.

## Tasks

- [ ] Select and configure the hosting target.
- [ ] Verify production data generation and static build behavior.
- [ ] Add only the project context needed by an external reader.
- [ ] Publish and document the canonical URL.

## Acceptance Criteria

- A clean checkout can build the production site reproducibly.
- The public site shows current canonical results without manual data refreshes.
- No W&B credentials, Modal credentials, or private artifacts are exposed.

## Source

[Lc0 Net](https://app.notion.com/p/35a8054d223580b79ebadc55321dd4d3)
