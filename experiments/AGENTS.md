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

Each model family has one programmable scaling recipe. For dense models,
`configs/dense.py` takes `d_model` as its scaling argument and derives the rest
of the baseline configuration. After an experimental run finishes:

1. Run `uv run compare-run WANDB_URL`.
2. If it reports `PROMOTE`, mark the previous run at that width stale and add the
   candidate to the best-runs file.
3. Regenerate the website scaling data.
4. Change the family recipe only when a multi-width experiment supports changing
   the scaling law, not for a one-off width result.

Best-run rows may set `stale = true` when they no longer represent the current
family tune. Tooling excludes stale rows from comparisons, fits, extrapolation,
and website data.

Experiment reports should contain the command, W&B URL, physical-FLOPs efficiency
multiplier, promotion verdict, and notable observations. Include the
modified-compute multiplier only for experiments about batch size, step count, or
score stability.
