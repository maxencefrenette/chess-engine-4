---
id: "01kzhkmx9s"
title: "Define the website narrative and information architecture"
status: completed
priority: high
effort: medium
parent: "01kzg0rdfm"
dependencies: []
tags: ["website", "planning", "user-driven"]
created_at: 2026-08-08
completed_at: 2026-08-08
---

# Define the website narrative and information architecture

## Objective

Turn the user's explanation of what the project is, why it exists, and what an
external reader should understand into an approved website brief. This is a
user-led ideation task, not an autonomous writing pass.

## Tasks

- [x] Create a separate interactive Codex thread when the user is ready to talk
  through the story in depth.
- [x] Identify the intended audiences, central thesis, and the questions the
  site must answer.
- [x] Decide the page hierarchy and the role of the existing scaling explorer.
- [x] List the experiments, diagrams, source links, and caveats needed as
  evidence.
- [x] Record explicit inclusions, exclusions, tone, and design preferences.
- [x] Produce a concise content and information-architecture brief for approval.

## Acceptance Criteria

- The user explicitly approves the narrative, audience, and page hierarchy.
- Every planned quantitative claim has an identified repository source.
- The brief separates core project documentation from optional technical posts.
- No implementation begins merely because the ideation thread was created.

## Approved Brief

Approved by the user on 2026-08-08.

### Narrative and audience

Chess Engine 4 is an end-to-end effort to produce a cost-efficient chess neural
network that works directly in lc0. The primary audience is technical builders,
with an opening that remains legible to curious chess and ML readers. The central
proof is the relationship between supervised training cost and expected policy
Elo; data engineering, scaling experiments, architecture work, and custom kernels
explain how the result is achieved.

### Page hierarchy

- **Overview:** a focused gateway led by the thesis and a simple
  training-cost-versus-policy-Elo chart, followed by a concise explanation,
  routes into the deeper pages, and selected findings.
- **How it works:** LCZero records to Rust/Parquet data to Modal training to
  checkpoint/Safetensors export to the custom lc0 backend to evaluation.
- **Architecture:** common inputs and heads, dense and MoE bodies, training and
  inference tradeoffs, and exact `N_total` versus `N_active` distinctions. Name
  the final model only after it has been selected.
- **Experiments:** curated findings first, followed by the preserved scaling
  explorer, family detail, methodology, and source experiment links.
- **Blog:** dated, authored technical essays with a blog-style index and strong
  individual article pages. Contextual links connect posts to Architecture, How
  it works, and Experiments.
- **Model:** do not publish a placeholder. Add the model card only after the
  `$20` run is validated, including measured results, a Safetensors checksum,
  download, limitations, and practical lc0 usage.

The repository README remains a concise project summary and link hub rather than
duplicating the website guide.

### Editorial workflow

Long-form technical posts must feel authored rather than generated in bulk.
Develop them one at a time through a user interview, source inspection, an
approved outline, and revision. Do not create launch filler or draft the initial
set in a single implementation pass. The initial post directions are:

- why a stacked MLP was chosen over a Transformer, and why MoE;
- how the scaling ladder was built;
- the dataset format and its engineering tradeoffs.

### Lead chart

- Use logarithmic present-day supervised reproduction cost on the x-axis and
  policy Elo from the common policy-only protocol on the y-axis.
- Include the exact four retained LCZero references: T74, T1, T3, and BT4.
- Distinguish measured Chess Engine 4 observations, the Chess Engine 4
  fit/extrapolation, and externally estimated reference-net costs.
- Keep the hero rendering simple and link to methodology containing confidence
  intervals, fit assumptions, cost ranges, and citations.
- Never describe policy Elo as searched engine strength.

### Visual direction

Use a technical-editorial system: warm ivory ground, ink typography, cobalt data
and navigation accents, restrained monospace labels, fine plot lines, and subtle
8-by-8 structure. Avoid chess-piece decoration and monitoring-dashboard chrome.

### Evidence map

- Chess Engine 4 run metrics: `experiments/best-runs-dense.toml` and
  `experiments/best-runs-moe64a2.toml`.
- Chess Engine 4 training cost: canonical recipes plus
  `experiments/throughput-dense.toml` and
  `experiments/throughput-moe64a2.toml`.
- Common policy-Elo field and raw evidence:
  `experiments/2026-08-07.01-dense-moe-policy-elo/`.
- Confidence-interval methodology:
  `experiments/2026-08-08.01-paired-elo-confidence/`.
- Architecture and parameter definitions: `configs/dense.py`,
  `configs/moe64a2.py`, and `src/chess_engine_4/model/`.
- Data scale: `experiments/training-data.toml`.
- LCZero reference-net training budgets and reproduction costs require a new,
  cited, reviewable input surface before they can appear publicly.

### Explicit exclusions

- Do not propose a canonical held-out evaluation dataset; the project trains for
  one epoch.
- Do not invent final-model metrics or identify a final architecture before the
  run exists.
- Do not require the exact training-code commit in the release presentation.
- Do not expose credentials, private artifact paths, or uncited estimates.
