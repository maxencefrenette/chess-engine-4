---
id: "01kzj7y1r1"
title: "Add SM120 dense training kernels"
status: in-progress
priority: high
effort: large
dependencies: []
tags: ["cuda", "sm120", "dense-training"]
created_at: 2026-08-08
---

# Add SM120 dense training kernels

## Objective

Fill the RTX PRO 6000 training gap by adding explicitly selected SM120 BF16
dense forward/backward kernels without changing canonical recipe selection or
the retained cuBLAS dense inference path.

## Tasks

- [ ] Add SM120 compilation, capability dispatch, and Python bindings.
- [ ] Preserve explicit-backend failure semantics for unsupported shapes and precisions.
- [ ] Add focused dispatch and numerical forward/backward tests.
- [ ] Compare the end-to-end custom layer against Transformer Engine on RTX PRO 6000.
- [ ] Run the repository verification gate and record the relevant pinned upstream commits.

## Acceptance Criteria

- Custom BF16 dense training resolves only for the exact SM120 capability and
  existing supported dense shapes/row alignment; unsupported precision, shape,
  or capability fails before launch and never falls back.
- The extension exposes SM120 BF16 GEMM, RMSNorm, SwiGLU, residual, and backward
  bindings with a device-side exact capability guard.
- Existing numerical thresholds remain enforced for forward output and all
  trainable gradients against Transformer Engine on an RTX PRO 6000.
- A focused paid job reports CUDA-graph forward and backward latency against
  Transformer Engine at the canonical benchmark shape; no training run is launched.
- `uv run pytest -q`, `uv run ruff check .`, `pnpm --dir website lint`,
  `pnpm --dir website build`, and `git diff --check` pass.
- The implementation records ThunderKittens `1c3920d993404dd49a6d4c7267ea11d583bd5c68`
  and Transformer Engine `8260f49660cbadb78bc52c90449428c51625469d`.

## Follow-up

Low-precision SM120 inference is out of scope until the current export/runtime
format can represent and validate it independently.
