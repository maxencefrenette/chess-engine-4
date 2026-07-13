# Experiments

This directory contains short experiment notes.

Keep these logs small. They should record the context needed to interpret a run
later: commit, config, command, data source, W&B URL, headline metrics, and any
notes about what changed or looked suspicious.

Do not commit raw W&B history, checkpoints, Modal logs, or large metric exports.

Name new experiment folders as `YYYY-MM-DD.NN-slug`, where `NN` is a
zero-padded sequence number for that day, for example
`2026-05-07.02-shape-down-batch`. This keeps same-day experiments sorted in
chronological order.

## Promotion protocol

Canonical model configs are named for residual-stream width, such as
`configs/dense/d128.toml`. After an experimental run finishes:

1. Run `uv run compare-run WANDB_URL`.
2. Promote the run as the new default for its width when its physical-FLOPs
   compute-efficiency multiplier is higher than the best incumbent multiplier for
   that same `d_model`.
3. Separately compare `loss_upper_1sd` with the fitted current-default curve at the
   candidate's derived modified compute: `flops_per_sample * batch_size * steps^2`.
4. Treat a negative residual as beating the global trend, not as a prerequisite
   for same-width promotion.
5. If promoted, update the matching width config and the best-runs data, regenerate
   scaling data, and record the command, W&B URL, modified compute, fitted score,
   residual, both compute-efficiency multipliers, and decision in the experiment
   report.

The same-width incumbent and cross-width trend answer different questions. Always
report both decisions when they disagree.

When reporting experiment results to the user, headline the compute-efficiency
multiplier from the `loss` versus physical-FLOPs fit. Prefer the
`loss_upper_1sd` versus modified-compute multiplier only when the experiment is
specifically about step allocation, batch efficiency, or score stability.
