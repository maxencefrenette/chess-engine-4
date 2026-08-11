# AdamH dense 16d learning-rate calibration

## Goal and method

Restore a separately calibrated AdamH learning-rate recipe for the adaptive
dense `16d` batch. Existing `0.055x` AdamH runs are retained as sweep cells;
new runs fill only missing `sqrt(2)`-spaced cells. All comparisons use seed 1,
the canonical data, losses, schedule, optimizer implementation, exact batch
`16d`, and matched accepted samples.

A compact law will be considered only if it selects the correct held-out grid
cell and stays within `0.005` final `loss/task[ema=0.99]`. Otherwise the recipe
will use an explicit width-indexed `16d` table. Selected learning rates must be
spike-free.

## Commands

```sh
uv run python experiments/2026-08-11.04-adamh-16d-learning-rates/launch.py \
  --stage bf16
uv run python experiments/2026-08-11.04-adamh-16d-learning-rates/launch.py \
  --stage bf16 --launch
uv run python experiments/2026-08-11.04-adamh-16d-learning-rates/launch.py \
  --stage bf16-edge --launch
uv run python experiments/2026-08-11.04-adamh-16d-learning-rates/launch.py \
  --stage mxfp8-lower
uv run python experiments/2026-08-11.04-adamh-16d-learning-rates/launch.py \
  --stage mxfp8-lower --launch
uv run python experiments/2026-08-11.04-adamh-16d-learning-rates/launch.py \
  --stage mxfp8-edge --launch
uv run python experiments/2026-08-11.04-adamh-16d-learning-rates/launch.py \
  --stage d1280-holdout --launch
uv run python experiments/2026-08-11.04-adamh-16d-learning-rates/launch.py \
  --stage d1280-edge --launch
uv run python experiments/2026-08-11.04-adamh-16d-learning-rates/launch.py \
  --stage d1280-edge2 --launch
uv run python experiments/2026-08-11.04-adamh-16d-learning-rates/launch.py \
  --stage canonical-adjusted --launch
uv run python experiments/2026-08-11.04-adamh-16d-learning-rates/launch_canonical_offarm.py
uv run python experiments/2026-08-11.04-adamh-16d-learning-rates/launch_canonical_offarm.py \
  --launch
```

## Results

The completed sweep rejects a shared batch multiplier or a smooth width law.
The BF16 selections imply `16d / 32d` LR ratios of approximately `0.70`,
`0.70`, `1.00`, and `1.00` at d64 through d512. In MXFP8, the maximum
spike-free cells are `0.0010`, `0.00044`, and `0.00044` at d768, d1024, and
d1280.

A power law fitted through the d768 and d1024 spike-free boundaries predicts
approximately `0.00023` at held-out d1280. The tested `0.00022` cell was stable
but reached `3.075932`; the stable `0.00031` cell reached `3.037491`, missing
the pre-registered `0.005` law gate by a wide margin. The stable `0.00044` cell
improved further to `2.998657`. The law is therefore rejected and an explicit
table is promoted.

| Width | Sweep-selected 16d LR | EMA loss | Spikes | EG_flops | W&B |
| ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 0.00500 | 3.882105 | 0 | 1.081x | [acj6chdv](https://wandb.ai/maxence-frenette/chess-engine-4/runs/acj6chdv) |
| 128 | 0.00350 | 3.637907 | 0 | 0.965x | [m49k7gdm](https://wandb.ai/maxence-frenette/chess-engine-4/runs/m49k7gdm) |
| 256 | 0.00500 | 3.376105 | 0 | 0.978x | [tiadc73r](https://wandb.ai/maxence-frenette/chess-engine-4/runs/tiadc73r) |
| 512 | 0.00350 | 3.127499 | 0 | 1.025x | [3ua9a9eg](https://wandb.ai/maxence-frenette/uncategorized/runs/3ua9a9eg) |
| 768 | 0.00100 | 3.073851 | 0 | 0.624x | [hhg6ggxn](https://wandb.ai/maxence-frenette/chess-engine-4/runs/hhg6ggxn) |
| 1024 | 0.00044 | 3.066344 | 0 | 0.366x | [cf2vylky](https://wandb.ai/maxence-frenette/chess-engine-4/runs/cf2vylky) |
| 1280 | 0.00044 | 2.998657 | 0 | 0.360x | [kos9zc2i](https://wandb.ai/maxence-frenette/chess-engine-4/runs/kos9zc2i) |

The MXFP8 selections are bracketed by ineligible higher cells: d768 `0.0011`
recorded one spike, d1024 `0.000625` recorded one spike, and d1280 `0.000625`
recorded one spike. At d512, both the higher `0.005` cell and lower `0.0025`
cell recorded one spike, leaving the existing spike-free `0.0035` retry.

## New run inventory and cost

This follow-up launched 21 new runs. Hardware cost computed from recorded W&B
runtime and repository GPU/CPU rates was approximately `$7.110`.

| Width | LR | EMA loss | Spikes | W&B |
| ---: | ---: | ---: | ---: | --- |
| 64 | 0.0025 | 3.905476 | 0 | [kg7ebzsb](https://wandb.ai/maxence-frenette/chess-engine-4/runs/kg7ebzsb) |
| 64 | 0.0050 | 3.882105 | 0 | [acj6chdv](https://wandb.ai/maxence-frenette/chess-engine-4/runs/acj6chdv) |
| 64 | 0.0071 | 3.888578 | 0 | [h0nxtsb5](https://wandb.ai/maxence-frenette/chess-engine-4/runs/h0nxtsb5) |
| 128 | 0.0025 | 3.643833 | 0 | [bja86oh9](https://wandb.ai/maxence-frenette/chess-engine-4/runs/bja86oh9) |
| 128 | 0.0050 | 3.638006 | 0 | [ew0g6d1v](https://wandb.ai/maxence-frenette/chess-engine-4/runs/ew0g6d1v) |
| 128 | 0.0071 | 3.641148 | 0 | [r7zrhsou](https://wandb.ai/maxence-frenette/chess-engine-4/runs/r7zrhsou) |
| 256 | 0.0025 | 3.384327 | 0 | [p29kks7y](https://wandb.ai/maxence-frenette/chess-engine-4/runs/p29kks7y) |
| 256 | 0.0071 | 3.380173 | 0 | [g605du2m](https://wandb.ai/maxence-frenette/chess-engine-4/runs/g605du2m) |
| 512 | 0.0025 | 3.144418 | 1 | [hj1ue28a](https://wandb.ai/maxence-frenette/chess-engine-4/runs/hj1ue28a) |
| 768 | 0.00090 | 3.083203 | 0 | [iwyrtqd6](https://wandb.ai/maxence-frenette/chess-engine-4/runs/iwyrtqd6) |
| 768 | 0.00100 | 3.073851 | 0 | [hhg6ggxn](https://wandb.ai/maxence-frenette/chess-engine-4/runs/hhg6ggxn) |
| 768 | 0.00110 | 3.065589 | 1 | [m10s6c9j](https://wandb.ai/maxence-frenette/chess-engine-4/runs/m10s6c9j) |
| 768 | 0.00125 | 3.055215 | 1 | [8vixjlhp](https://wandb.ai/maxence-frenette/chess-engine-4/runs/8vixjlhp) |
| 1024 | 0.00031 | 3.104243 | 0 | [si9wos95](https://wandb.ai/maxence-frenette/chess-engine-4/runs/si9wos95) |
| 1024 | 0.00044 | 3.066344 | 0 | [cf2vylky](https://wandb.ai/maxence-frenette/chess-engine-4/runs/cf2vylky) |
| 1024 | 0.000625 | 3.029752 | 1 | [689epdqe](https://wandb.ai/maxence-frenette/chess-engine-4/runs/689epdqe) |
| 1024 | 0.00090 | 2.990422 | 1 | [a0n9o0d8](https://wandb.ai/maxence-frenette/chess-engine-4/runs/a0n9o0d8) |
| 1280 | 0.00022 | 3.075932 | 0 | [3fr9w2od](https://wandb.ai/maxence-frenette/chess-engine-4/runs/3fr9w2od) |
| 1280 | 0.00031 | 3.037491 | 0 | [mwiluhzf](https://wandb.ai/maxence-frenette/chess-engine-4/runs/mwiluhzf) |
| 1280 | 0.00044 | 2.998657 | 0 | [kos9zc2i](https://wandb.ai/maxence-frenette/chess-engine-4/runs/kos9zc2i) |
| 1280 | 0.000625 | 2.960140 | 1 | [rjlufjwg](https://wandb.ai/maxence-frenette/chess-engine-4/runs/rjlufjwg) |

Historical cells reused from
`experiments/2026-08-11.03-hyperball` provide the remaining bracket points and
are not included in the `$7.110` follow-up cost.

## Manual stability margin

After reviewing the completed curves, the user judged the spike-free d256 and
d512 selections too aggressive: absence of a detected spike in a single run is
not strong evidence that the rate is robust. The final recipe therefore uses a
log-log interpolation between the retained neighboring 16d anchors, d128 at
`0.0035` and d768 at `0.001`:

| Width | Sweep selection | Final recipe | Basis |
| ---: | ---: | ---: | --- |
| 256 | 0.0050 | 0.0022 | Rounded log-log interpolation. |
| 512 | 0.0035 | 0.0013 | Rounded log-log interpolation. |

These interpolated overrides were not part of the calibration sweep and must
not be cited as measured optima; their subsequent canonical reruns are recorded
below.

## Adjusted canonical-cell reruns

The final interpolated d256 and d512 recipe cells were subsequently rerun at the
canonical `0.055x` horizon. Both used exact batch `16d`, seed 1, and matched the
existing accepted sample counts. Both were spike-free, but neither passed the
`best-runs-dense.toml` promotion rule:

| Width | LR | EMA loss | Spikes | EG_flops | Incumbent EG | Verdict | W&B |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 256 | 0.0022 | 3.388976 | 0 | 0.911x | 0.989x | Canonical recipe override | [xljjmfxc](https://wandb.ai/maxence-frenette/chess-engine-4/runs/xljjmfxc) |
| 512 | 0.0013 | 3.184742 | 0 | 0.721x | 1.029x | Canonical recipe override | [97mwm56n](https://wandb.ai/maxence-frenette/chess-engine-4/runs/97mwm56n) |

The two reruns cost approximately `$0.130` from recorded W&B runtimes and
repository hardware rates, bringing new work in this experiment to about
`$7.240`. Although both reruns fail the ordinary higher-EG promotion rule, the
user explicitly designated them canonical because the registry should represent
the current conservative recipe rather than the superseded higher-LR recipe.

The scaling fit also contained d256/0.1x and d512/0.1x off-arm observations that
select batch `16d`; these were therefore rerun with the active table as well:

| Width | Ratio | LR | EMA loss | Spikes | EG_flops | Prior EG | W&B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 256 | 0.1 | 0.0022 | 3.289253 | 0 | 0.965x | 1.017x | [gq0xgwl1](https://wandb.ai/maxence-frenette/chess-engine-4/runs/gq0xgwl1) |
| 512 | 0.1 | 0.0013 | 3.091250 | 0 | 0.767x | 1.006x | [5vcqk48c](https://wandb.ai/maxence-frenette/chess-engine-4/runs/5vcqk48c) |

They cost approximately `$0.209`, bringing new experiment work to about
`$7.449`. By explicit user direction they also replace the superseded higher-LR
rows, making every d256/d512 scaling-fit observation that selects `16d`
consistent with the active conservative recipe. Their 0.2x and 0.3x rows select
`32d` and remain unchanged.

## Verdict

Use the explicit `16d` AdamH LR table, including the manual stability overrides,
while retaining the existing width-indexed `32d` table. The Hyperball paper
motivates improved LR transfer across model width and depth, but it does not
establish batch-size invariance; the project's matched results show that neither
a shared batch multiplier nor a simple width law is adequate here. The adjusted
d256/d512 cells are stable but intentionally trade away measured FLOP efficiency;
they replace the higher-LR registry rows by explicit user direction so canonical
scaling evidence remains aligned with the active recipe.
