---
id: "01kzj7y1r1"
title: "Add SM120 dense training kernels"
status: completed
priority: high
effort: large
dependencies: []
tags: ["cuda", "sm120", "dense-training"]
created_at: 2026-08-08
completed_at: 2026-08-08
---

# Add SM120 dense training kernels

## Objective

Fill the RTX PRO 6000 training gap by adding explicitly selected SM120 BF16
dense forward/backward kernels without changing canonical recipe selection or
the retained cuBLAS dense inference path.

## Tasks

- [x] Add SM120 compilation, capability dispatch, and Python bindings.
- [x] Preserve explicit-backend failure semantics for unsupported shapes and precisions.
- [x] Add focused dispatch and numerical forward/backward tests.
- [x] Compare the end-to-end custom layer against Transformer Engine on RTX PRO 6000.
- [x] Run the repository verification gate and record the relevant pinned upstream commits.

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

## Result

The BF16 path passed numerical checks on RTX PRO 6000. The final matched d256,
batch-8192 benchmark measured a 4.6355 ms custom synthetic step versus 3.6488 ms
for Transformer Engine, and a 7.4324 ms custom real-pipeline step versus 6.3739
ms for Transformer Engine. The backend therefore remains explicit-only and was
not promoted. See `experiments/2026-08-08.02-sm120-dense-training/README.md`.

Verification passed with 141 tests, Ruff, website lint/build, and diff check.
