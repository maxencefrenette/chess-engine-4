---
id: "01m0vr23hw"
title: "Prove that phase-routed model sparsity beats one dense model"
status: in-progress
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

The first sparse-system pilot consists of:

- Two independently trained d1024 models initialized from scratch.
- A piece-count boundary that routes every position to exactly one specialist.
- A manifest that assigns positions on either side of the boundary to the
  corresponding specialist.
- Leaf-routed lc0 inference: every evaluated position is sent to the model named
  by the manifest, including when one inference batch contains positions for
  both models.

The comparison must hold total training compute fixed. The two specialists each
receive D accepted positions and the single-model control receives 2D accepted
positions. At inference, both systems activate only one d1024x8 model for each
leaf.

## Experiment

- Use the data-selected boundary of at least 20 pieces versus at most 19 pieces;
  a 122,880-position, deep-offset, shard-stratified sample measured a
  50.37%/49.63% split.
- Run the initial low-cost crossover probe at 0.05x Chinchilla per specialist:
  about 275 million accepted positions for each specialist versus about 550
  million positions for one general d1024 control.
- Declare the total training-compute budget and its allocation before launching
  either arm.
- Train both specialists from scratch and train the equal-compute general
  control from scratch.
- Keep runtime sampling at 1.0 and use temporary on-the-fly piece-count
  filtering for the pilot.
- Let the general control use the natural all-position stream. The measured
  50.37%/49.63% phase mixture is close enough to balanced without reconstructing
  batches in Python; report the realized phase fractions with its regional loss.
- Compare smoothed one-pass training loss. Record the general model's loss
  separately on both sides of the boundary as well as its aggregate loss.
- If the training-loss pilot succeeds, create a minimal versioned manifest
  mapping each side of the boundary to its specialist.
- Implement two-model leaf routing in the project-owned lc0 backend. Keep both
  complete models resident in VRAM, partition mixed inference batches by route,
  and restore outputs to their original order.
- Benchmark the routed backend at an aggregate batch size around 1,024 and
  verify that both d1024x8 MXFP8 models and inference working memory fit on the
  target local GPU.
- If the training-loss pilot succeeds, compare regional output calibration and
  run a paired Elo tournament between the sparse two-model system and the
  equal-compute single-model control.

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
- The routed combination of specialist training losses beats the general
  control's equal-compute training loss, or the pilot records a negative result.
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
- Regional loss and output-calibration comparison between the general model and
  both specialists.
- Paired `eval-tournament-modal` Elo comparison at identical lc0 settings.
- Repository-wide verification required by `AGENTS.md` before broad commits.

## Long-Term Vision

If the fixed-compute two-model experiment succeeds, repeat the process to grow a
manifest-routed sparse ensemble. The eventual direction may include roughly 32
d1024x8 models routed by piece count and other relatively stable features such
as queen presence. That scale-up is future work and is not required to complete
this task.

## Progress

- 2026-09-07: Completed the from-scratch `D = 0.05x` pilot at the 20-piece
  cutoff. Both specialists lost to the equal-compute general control on their
  own regions. The sampled-mixture routed EMA was 3.02380 versus 3.00368 for
  the control's regional EMAs, so this point establishes `D* > 0.05x` if a
  crossover exists. Full commands, compute accounting, W&B URLs, runtime
  observations, and the negative verdict are recorded in
  `experiments/2026-09-07.01-phase-routed-d1024-pilot/`.
- The negative pilot does not justify building the manifest/router or running
  Elo.
- 2026-09-07: Repeated the matched test at `D = 0.1x` per specialist versus a
  canonical `0.2x` general control. The routed loss was 2.92949 versus 2.78413
  from the control's weighted regional EMAs, a 0.14535 (5.22%) deficit. Both
  specialists again lost in their own regions. Details are in
  `experiments/2026-09-07.01-phase-routed-d1024-pilot/` together with the first
  point.
- No crossover is observed at either `D = 0.05x` or `D = 0.1x`. The canonical
  ratio-specific batch and learning-rate recipes remain part of the comparison.
- 2026-09-07: Tested d768 at `D = 0.2x` per specialist versus a `0.4x`
  equal-compute general control. The routed loss was 2.78733 versus 2.78793,
  a nominal 0.00060 (0.02%) advantage. This is the first non-negative point,
  but it is too close to call from one seed and does not directly locate the
  d1024 crossover because model width changed.
- 2026-09-07: Tested d512 at `D = 0.5x` per specialist versus a `1.0x`
  equal-compute general control. Both specialists won in-region; routed loss
  was 2.80516 versus 2.81807, an improvement of 0.01291 (0.46%). This is the
  first clearly positive point, but it requires replication and does not
  directly locate the d1024 crossover because width changed.
