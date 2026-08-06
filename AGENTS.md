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
- Training is B200-only and uses Transformer Engine. Evaluation may use cheaper
  Modal GPUs when the selected backend supports them.
- Review the printed launch summary before allowing an expensive run to proceed.
- Use Modal concurrency for independent sweeps, while respecting the user's
  stated dollar and wall-clock budgets.

## Experiments

- Keep the loss configuration fixed unless the training target is under study.
- Compare completed candidates with `uv run compare-run WANDB_URL`.
- Promote a candidate at an existing width only when its `EG_flops` exceeds the
  incumbent and it has no detected loss spikes, unless the experiment explicitly
  optimizes another objective such as realized training cost. In that case,
  report the `EG_flops` tradeoff and select against the stated objective.
- Update `experiments/best-runs-dense.toml` only after promotion, then regenerate
  website data with `uv run export-scaling-data`.
- Preserve historical experiment reports. Do not rewrite old terminology or
  delete old results merely because the current methodology changed.
- Follow `experiments/AGENTS.md` when adding a report.

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
