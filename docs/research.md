# Research analysis and planning

Research tools live in `src/chess_engine_4/analysis/`. Their CLI names and
numerical behavior are unchanged; moving them does not discard experiment results.
Offline planning and website-data generation do not require Modal. The
`throughput-sweep` execution path still uses [Modal](cloud.md).

The budget planners use recorded cloud throughput and dollar-cost assumptions,
not measured RTX 5070 wall-clock time or a local VRAM-fit guarantee. Use them
for their original research purpose, not as local launch authorization.

Plan the predicted lowest-validation-loss dense run with:

```sh
uv run plan-budget 5 10 100
```

Plans default to 25 billion available positions. Use `--assume-samples N` to
evaluate another data budget, or `--dataset experiments/training-data.toml` to
use the recorded corpus size. Add `--families dense moe64a2` to compare both
families. Loss predictions use the curated Skaling `L(N,D)` observations.
Training horizon is capped at twice the largest family-wide observed ratio.
Dense widths may be extrapolated through d2560; their cost holds the largest
measured width's MFU constant and scales wall time by FLOPs per step. Such
costs remain explicitly labeled as width extrapolations.
Predicted losses include deterministic 80% bootstrap intervals. Compare cheap preliminary
runs against spending the same fixed total budget directly on final training with:

```sh
uv run plan-budget 10 --suggest-runs 3
```

Use `--focus-budget` and `--max-suggestion-cost` to change the targeted breakpoint and
per-run cost ceiling. Preliminary-run cost is deducted from the fixed focus budget, and
the preliminary checkpoint is treated as sunk rather than reusable final training.

Scaling-law acquisition uses the declared L-shaped dense grid. Complete its d64
data arm and `0.055x` width arm before ranking cheap, unobserved interior cells:

```sh
uv run plan-ladder --focus-budget 10 --count 3
```

Recorded cells count toward scaffold coverage even when they contain a loss spike;
the spike remains available separately for fit and promotion decisions.


## Evidence and methodology

Keep historical reports in `experiments/`; retain both model families and their
registries. Promotion and experiment rules are defined in [AGENTS.md](../AGENTS.md).
[OPTIMIZATION.md](../OPTIMIZATION.md) records training implementation choices.

The dashboard generator consumes those registries and recipes. Its generated
JSON is disposable; the reports and TOML evidence are not. See
[website development](../website/README.md) for the watcher and build workflow.
