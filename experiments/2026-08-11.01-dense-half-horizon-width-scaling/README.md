# Dense half-horizon width scaling

## Goal

Test whether the adaptive `16d` dense recipe can identify model-width scaling
at half the previous `0.1x` profiling horizon.

Exact `0.05x` preflight passes only d64 and d128. D256 through d1024 are
rejected by the fitted minimum-step rule, missing it by 0.7% to 5.7%. The
closest common round ratio that clears every width is `0.055x`, or 55% of the
old horizon. It uses `16d` at every width and is tested without overriding the
production guardrail.

No result was promoted before review. After the successful review, the six
stable selected rows were added to the canonical scaling-law evidence.

## Preflight

Exact `0.05x` dry runs selected `16d` at d64 and d128, but the production
recipe rejected d256 through d1024:

| Width | `16d` steps at `0.05x` | Fitted minimum | Minimum accepted ratio |
| ---: | ---: | ---: | ---: |
| d64 | 2,392 | 1,804 | `0.03770x` |
| d128 | 3,348 | 3,154 | `0.04707x` |
| d256 | 5,268 | 5,519 | `0.05238x` |
| d512 | 9,108 | 9,658 | `0.05302x` |
| d768 | 12,946 | 13,398 | `0.05174x` |
| d1024 | 16,786 | 16,900 | `0.05034x` |

The experiment therefore uses the smallest clean common ratio above every
boundary, `0.055x`. This is 55% of the previous `0.1x` width arm rather than an
exact half. The six-run launch summary estimated `$1.244` conservatively.

## Runs

The source checkout was based on `f76410931a62649f79188d81659669dec441d2f8`
with the reviewed adaptive dense recipe present in the worktree. All runs used
seed 1, the canonical loss and schedule, and automatic batch, step, and LR
selection:

```sh
uv run python experiments/2026-08-11.01-dense-half-horizon-width-scaling/launch.py
uv run python experiments/2026-08-11.01-dense-half-horizon-width-scaling/launch.py --launch
```

| Width | Batch | Steps | EMA loss | EG_flops | Spikes | Cost | W&B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| d64 | 1,024 | 2,630 | 3.889359 | 1.794x | 0 | $0.010 | [adt7rfr5](https://wandb.ai/maxence-frenette/chess-engine-4/runs/adt7rfr5) |
| d128 | 2,048 | 3,684 | 3.668101 | 0.926x | 0 | $0.013 | [6avq3le8](https://wandb.ai/maxence-frenette/chess-engine-4/runs/6avq3le8) |
| d256 | 4,096 | 5,794 | 3.409629 | 0.695x | 0 | $0.023 | [ufjguuuy](https://wandb.ai/maxence-frenette/chess-engine-4/runs/ufjguuuy) |
| d512 | 8,192 | 10,018 | 3.173032 | 0.637x | 1 | $0.104 | [t2yqwy91](https://wandb.ai/maxence-frenette/chess-engine-4/runs/t2yqwy91) |
| d768 | 12,288 | 14,242 | 3.030408 | 0.913x | 1 | $0.203 | [h568jgua](https://wandb.ai/maxence-frenette/chess-engine-4/runs/h568jgua) |
| d1024 | 16,384 | 18,466 | 2.938255 | 1.354x | 1 | $0.439 | [1vrolagz](https://wandb.ai/maxence-frenette/chess-engine-4/runs/1vrolagz) |

Recorded hardware cost was `$0.791`, including the repository GPU and CPU
rates. The d512, d768, and d1024 spikes occurred at steps 5,960, 9,250, and
15,780. Their EMA losses subsequently improved to the final values above.

## Lower-LR retries

The three spiked widths were rerun with the next lower explicit LR multiplier.
Every other configuration field and accepted sample count remained matched:

| Width | Multiplier | LR | EMA loss | EG_flops | Spikes | Cost | W&B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| d512 | 0.85x | 0.000493 | 3.168128 | 0.675x | 0 | $0.086 | [u4c69pip](https://wandb.ai/maxence-frenette/chess-engine-4/runs/u4c69pip) |
| d768 | 1.00x | 0.000330 | 3.033531 | 0.872x | 0 | $0.272 | [gk7to9xk](https://wandb.ai/maxence-frenette/chess-engine-4/runs/gk7to9xk) |
| d1024 | 1.15x | 0.000253 | 2.941948 | 1.268x | 0 | $0.429 | [eaghg0z2](https://wandb.ai/maxence-frenette/chess-engine-4/runs/eaghg0z2) |

All three lower LRs are stable and are promoted into the dense recipe. The
retry cost was `$0.787`, making the complete nine-run experiment `$1.579`.

## Scaling-law test

The test keeps the established d64 data arm and replaces only the model-size
arm. Both fits predict the same seven canonical off-arm observations at d128,
d256, and d512. The matched comparison excludes d768 because the old arm did
not contain that width.

| Width arm | Held-out MAPE | Model exponent | Coupling | Effective exponent `alpha*k` |
| --- | ---: | ---: | ---: | ---: |
| Previous `0.1x` | 1.221% | 0.738 | 0.159 | 0.118 |
| New `0.055x`, matched widths | **0.736%** | 1.103 | 0.082 | 0.091 |
| New `0.055x`, including d768 | **0.725%** | 1.110 | 0.083 | 0.092 |
| New stable rows only | **0.725%** | 1.110 | 0.083 | 0.092 |

The stable short arm reduces held-out error by 41%. Its leave-one-width-out
errors are below `0.29%`, while the effective model exponent remains in
`[0.088, 0.092]`. The canonical 19-row fit predicts the six selected losses
with `0.835%` MAPE. The shorter arm contains coherent width-scaling information.

Reproduce with:

```sh
uv run python experiments/2026-08-11.01-dense-half-horizon-width-scaling/analyze.py
```

## Verdict

**Success at the minimum safe common horizon.** With the lower promoted LRs,
all six `0.055x` width points are stable. The resulting L-shaped Skaling fit
reduces held-out MAPE from `1.221%` to `0.725%` while using 55% of the previous
width-arm samples. This validates the adaptive batch recipe for cheap model-size
profiling, although the fitted minimum-step guard still rejects an exact
`0.05x` arm at four widths.

After review, the stable observations were promoted to the canonical
`scaling_runs` registry alongside the LR dictionary entries. The original
spiked attempts remain historical observations in this report.

## Methodology

The constant-ratio cross-scale ladder follows *MAI-Thinking-1*. The sparse
d64 data arm plus low-horizon width arm follows the L-shaped profiling strategy
from *Skaling*. Those papers motivate the test structure; all thresholds, LR
choices, losses, and verdicts above are project measurements.
