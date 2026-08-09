---
id: "01kzj8kfvz"
title: "Refresh the training and kernel toolchain"
status: blocked
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

## Blocker

Wait for the remaining SM90 kernel work to merge.
