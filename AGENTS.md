# Chess Engine 4

This repository trains LCZero-compatible networks on Modal. The Python package
at the repository root is the primary project; `website/` is a static dashboard
for canonical experiment results.

## Training

- Use `uv run train-modal`; local training is not supported.
- The canonical dense recipe is `configs/dense.py`.
- Routine experiments default to `0.2x` Chinchilla. Specify another training
  ratio only when data allocation is the variable under study or for a planned
  final run.
- Review the printed launch summary before allowing an expensive run to proceed.
- Dollar amounts for runs are targets, not hard caps. If a run lightly exceed its budget, keep it going.

## Experiments

- Keep the loss configuration fixed unless the training target is under study.
- Compare completed candidates with `uv run compare-run WANDB_URL`.
- Promote a candidate at an existing width only when its `EG_flops` exceeds the
  incumbent.
- Tune learning rate using only runs without loss spikes; a selected learning
  rate must be spike-free. In experiments tuning other parameters, loss spikes
  do not disqualify the parameter result: record the spikes and whether training
  returned to trend, then retune learning rate for the selected configuration.
- Update `experiments/best-runs-dense.toml` only after promotion.
- Preserve historical experiment reports. Do not rewrite old terminology or
  delete old results merely because the current methodology changed.
- Follow `experiments/AGENTS.md` when adding a report.

## Tasks

Use taskmd for durable work that spans multiple commits, has dependencies, or
should remain in the project backlog. Do not create tasks for incidental edits.

- Run `taskmd next` before selecting backlog work.
- Set a task to `in-progress` before starting it.
- Respect its dependencies, acceptance criteria, and declared verification.
- Set it to `completed` only after the implementation and verification finish.
- Do not create separate task worklogs; keep relevant progress in the task file.
- Keep permanent rules in `AGENTS.md` and completed experimental evidence in
  `experiments/`; task files track work, not results.

## Verification

For kernel work, follow `.agents/skills/kernel-development/SKILL.md` and inspect the local
upstream references before introducing a new implementation pattern.
For ML architecture or methodology work, follow `.agents/skills/ml-references/SKILL.md` and
consult the local paper manifest.

Run the relevant focused tests while iterating. Before committing a broad
change, run:

```sh
uv run pytest -q
uv run ruff check .
pnpm --dir website lint
pnpm --dir website build
git diff --check
```
