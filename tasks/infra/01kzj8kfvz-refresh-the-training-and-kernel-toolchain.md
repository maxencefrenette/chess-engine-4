---
id: "01kzj8kfvz"
title: "Refresh the training and kernel toolchain"
status: pending
priority: medium
effort: large
dependencies: ["01kzhp5w47"]
tags: ["dependencies", "python", "cuda", "pytorch", "transformer-engine", "thunderkittens", "rust", "website"]
created_at: 2026-08-08
---

# Refresh the training and kernel toolchain

## Objective

Upgrade the project toolchain as far as is useful without breaking correctness,
reproducibility, or performance.

## Tasks

- [ ] Update Python, uv dependencies, PyTorch, Transformer Engine, Modal/CUDA,
      ThunderKittens, Rust, and website tooling in compatible stages.
- [ ] Refresh pins and locks; briefly record any deferred upgrades and why.
- [ ] Recheck representative training, kernels, export, and LCZero inference.

## Acceptance Criteria

- Clean checkouts remain reproducible and the full repository gate passes.
- No numerical or material performance regression is retained.

## Progress

- 2026-10-11: Removed the obsolete SM90 blocker; dependency `01kzhp5w47` is
  completed. The website dependency refresh is tracked in `01m4p21n3g`.
- Training/kernel upgrades remain pending until native builds, numerical checks,
  and performance verification can run without competing with Urban-rivals-ai.
  Keep uv on `latest`, as requested; do not change the Porygon Docker image.
- Website follow-up: the 2026-10-11 full pnpm audit still reports
  [GHSA-vfj7-8cjw-p6xm](https://github.com/advisories/GHSA-vfj7-8cjw-p6xm)
  in the development-only ESLint → fast-glob → micromatch → braces chain, with
  no patched version reported. Recheck upstream before the next website refresh;
  the production-only audit is clean. No forced override was added.
