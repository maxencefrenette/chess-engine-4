---
id: "01m0vr23hw"
title: "Prove that phase-routed model sparsity beats one dense model"
status: pending
priority: medium
effort: large
dependencies: []
tags: ["research", "training", "inference", "sparsity", "lc0"]
created_at: 2026-08-24
---

# Prove that phase-routed model sparsity beats one dense model

## Objective

Test whether, for a fixed total training-compute budget, it is better to train
two small dense models and route positions between them than to spend the entire
budget on one dense model.

The sparse system consists of:

- A base model trained on all phases of chess.
- A copy of the base model fine-tuned to outperform it on one selected game
  phase.
- A manifest that assigns positions in that phase to the fine-tuned model and
  all other positions to the base model.
- Leaf-routed lc0 inference: every evaluated position is sent to the model named
  by the manifest, including when one inference batch contains positions for
  both models.

The comparison must hold total training compute fixed. Compute used to train the
base and fine-tune the second model counts toward the sparse system's budget. The
single-model control receives the same total training compute. At inference,
both systems activate only one d1024x8 model for each leaf.

## Experiment

- Analyze t80 positions by piece count and select one promising phase boundary
  from the data. The illustrative ranges discussed so far, such as 32--22 and
  21--16 pieces, are not predetermined choices.
- Declare the total training-compute budget and its allocation before launching
  either arm.
- Train the all-chess base model.
- Spend the sparse arm's remaining budget fine-tuning one copy of the base on
  the selected phase.
- Spend the control arm's corresponding remaining budget continuing to improve
  one general model on all-chess data.
- Create a minimal versioned manifest mapping the selected phase to the
  fine-tuned model and everything else to the base model.
- Implement two-model leaf routing in the project-owned lc0 backend. Keep both
  complete models resident in VRAM, partition mixed inference batches by route,
  and restore outputs to their original order.
- Benchmark the routed backend at an aggregate batch size around 1,024 and
  verify that both d1024x8 MXFP8 models and inference working memory fit on the
  target local GPU.
- Compare regional validation metrics and run a paired Elo tournament between
  the sparse two-model system and the equal-compute single-model control.

## Acceptance Criteria

- The single-model and two-model arms use the same declared total training
  compute budget.
- The phase interval is selected from recorded piece-count, data-volume, and
  base-model-quality analysis rather than assumed in advance.
- The manifest deterministically routes every supported position to exactly one
  of the two checksum-verified models.
- lc0 performs true leaf routing inside mixed batches and returns outputs in the
  original batch order.
- Both models remain resident in VRAM; model loading is not on the inference
  critical path.
- The phase model improves the selected region over the base model.
- At identical search settings, the complete routed system beats the
  equal-training-compute single-model control in a paired Elo tournament.
- The experiment report records the compute accounting, commands, W&B URLs,
  manifest, model artifacts, regional metrics, backend benchmarks, Elo result,
  and verdict.

## Verification

- Focused tests for manifest validation, deterministic routing, mixed-batch
  partitioning, and output scattering.
- Exact-output comparison when both manifest routes point to the same model.
- Resident VRAM, latency, throughput, and routed sub-batch measurements on the
  target local GPU at representative batches up to 1,024 positions.
- Regional loss and output-calibration comparison between the base and phase
  model.
- Paired `eval-tournament-modal` Elo comparison at identical lc0 settings.
- Repository-wide verification required by `AGENTS.md` before broad commits.

## Long-Term Vision

If the fixed-compute two-model experiment succeeds, repeat the process to grow a
manifest-routed sparse ensemble. The eventual direction may include roughly 32
d1024x8 models routed by piece count and other relatively stable features such
as queen presence. That scale-up is future work and is not required to complete
this task.
