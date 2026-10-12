# Research analysis

Supported analysis lives in `src/chess_engine_4/analysis/`:

- Skaling fits over the retained canonical observations.
- `uv run compare-run WANDB_URL` to compare completed runs with the frontier.
- `uv run generate-website-data` to regenerate the public dashboard payload.
- `uv run --extra modal throughput-sweep --help` for deliberate cloud throughput
  measurements. Offline cache helpers do not require the Modal SDK.

The dollar-budget optimizer and L-shaped ladder planner have been retired.
`plan-budget` and `plan-ladder` are no longer installed commands. Their old grids,
throughput caches, and reports remain historical evidence; see
[historical reproduction](history.md) for the last implementation revision.
This does not remove scaling fits, run comparison, or website loss curves.

Canonical recipes describe the model and training budget, not cloud placement.
A recipe's GPU remains unresolved until a launcher selects local hardware or
applies the Modal placement profile. Existing cloud choices are retained in
`backends/modal/deployment.py`; they are not local throughput or VRAM guarantees.

## Evidence and methodology

Keep historical reports in `experiments/`; retain both model families and their
registries. Promotion and experiment rules are defined in [AGENTS.md](../AGENTS.md).
[OPTIMIZATION.md](../OPTIMIZATION.md) records training implementation choices.

The dashboard generator consumes the registries and recipes. Its generated JSON
is disposable; reports and TOML evidence are not. See
[website development](../website/README.md) for the watcher and build workflow.

Some older analysis scripts intentionally require a historical checkout.
[The script manifest](../experiments/historical-scripts.toml) records a compatible
revision; do not restore retired fitting laws in the active package to run them.
