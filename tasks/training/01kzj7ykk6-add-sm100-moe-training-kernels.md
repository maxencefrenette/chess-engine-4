---
id: "01kzj7ykk6"
title: "Add SM100 MoE training kernels"
status: completed
priority: high
effort: large
dependencies: []
tags: ["kernels", "sm100", "moe"]
created_at: 2026-08-08
---

# Add SM100 MoE training kernels

## Objective

Add explicit SM100/B200 training support for the existing sorted, padded BF16
`moe64a2` expert implementation without changing canonical recipe selection.
Preserve the supported topology (64 experts, top-2, expansion ratio 2) and
widths (128, 256, 512), and reject every unsupported custom configuration.

## Tasks

- [x] Audit existing SM100 dense training and standalone BF16 dense/MoE inference.
- [x] Inspect pinned Mixture-of-Kittens, Transformer Engine, and CUTLASS sources.
- [x] Add SM100 forward, trainable forward, backward, bindings, and capability dispatch.
- [x] Add focused dispatch and numerical correctness coverage.
- [x] Benchmark layer and matched end-to-end training-step performance on B200.
- [x] Run the repository verification gate and commit separable changes.

## Acceptance Criteria

- SM100 custom BF16 MoE accepts only d128/d256/d512, 64 experts, top-2,
  expansion ratio 2, SwiGLU, and aligned rows; unsupported configurations fail.
- Forward output cosine similarity is at least 0.999 against a BF16 PyTorch
  reference; every trainable gradient has cosine similarity at least 0.99 and
  contains only finite values.
- Focused correctness and timing run on an actual Modal B200.
- Custom and Transformer Engine are compared with matched model, precision,
  batch, optimizer, and synthetic training step settings on B200.
- Transformer Engine remains canonical unless custom wins the matched
  end-to-end training-cost comparison.
- `uv run pytest -q`, `uv run ruff check .`, `pnpm --dir website lint`,
  `pnpm --dir website build`, and `git diff --check` pass.
