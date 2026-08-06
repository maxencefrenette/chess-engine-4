---
name: kernel-development
description: Develop, optimize, benchmark, or review CUDA, ThunderKittens, Transformer Engine, dense, or MoE kernels in chess-engine-4. Use for kernel architecture, fusion, precision, dispatch, scheduling, and performance work.
---

# Kernel Development

Run `uv run python reference/sync.py` when `reference/repos/` is absent or current upstream
code matters. Read `reference/repos.toml`, then inspect only the repositories relevant to the
task. ThunderKittens is already available at `third_party/ThunderKittens`.

Prefer upstream implementation evidence over assumptions. Preserve this project's numerical
acceptance checks and benchmark against the canonical end-to-end training path. Record the
upstream repository and commit when a design or report materially depends on reference code.
