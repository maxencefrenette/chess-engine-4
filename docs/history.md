# Historical scripts and retired features

Experiment reports and result files are evidence, not a promise that every old
analysis script runs with today's Python package. Preserve the evidence; use a
matching historical checkout to rerun code that depends on retired APIs.

## Revision-bound analysis

The three scripts in [historical-scripts.toml](../experiments/historical-scripts.toml)
depend on the old FLOPs-only and undertraining loss laws. Those APIs were removed
by the Skaling cutover (`5fb2810`), before the local-first package reorganization.
Renaming their imports alone does not restore compatibility. Do not run the
copies in the current checkout or add obsolete fitting APIs back to production.

The manifest records full revision
`5c83fe26fd1ce2721e1c6165d1766d436094e582`, immediately before that cutover.
The scripts, input files, and required fitting definitions were checked in Git
at this revision. It is an API-compatible reference snapshot, not a claim that
every original figure was produced at that exact commit. Chart regeneration
was not run as part of this maintenance pass.

To reproduce in a separate checkout without disturbing current work:

```sh
git worktree add --detach ../chess-engine-4-history 5c83fe26fd1ce2721e1c6165d1766d436094e582
cd ../chess-engine-4-history
mise install
uv sync --dev
uv run python experiments/2026-07-12.08-geglu-canonical-cutover/plot.py
```

Use the other commands from the manifest in that same historical checkout.
Scripts may overwrite their generated figures there; inspect the worktree diff
without copying old output over current evidence. Historical dependency setup
and plotting consume CPU, so schedule them separately from production training.

## Retired production features

- `plan-budget` and `plan-ladder`, their implementation, and their dedicated tests
  were removed. The last version is at `fd16498`; historical scaling grids and
  throughput measurements remain. Run them only in a checkout of that revision
  if reproducing old cloud-allocation decisions. Current scaling fits,
  `compare-run`, throughput collection, and the dashboard remain supported.
- NVFP4 training was never promoted and is no longer a training option. Its
  [inference investigation](../experiments/2026-08-11.03-nvfp4-inference-crossover/README.md)
  remains intact. This does not remove engine inference support.
- Production training is SwiGLU-only, matching our custom lc0 export path.
  Portable reference models and historical activation metadata remain for
  interpreting older checkpoints and experiments; they are not supported new
  training configurations.
