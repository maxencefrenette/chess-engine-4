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
