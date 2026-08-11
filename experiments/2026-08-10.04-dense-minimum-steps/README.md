# Dense Minimum Optimizer Steps

## Goal

Calibrate the dense learning-rate adjustment from `B=32d` to `B=16d`, then
establish the conservative minimum number of optimizer steps required for the
dense recipe. The intended selector is:

```text
use B=32d if its step count clears the validated minimum
otherwise use B=16d if its step count clears the validated minimum
otherwise reject the configuration
```

The experiment has an authorized `$8` hardware ceiling and targets less than
`$5`. Every stage is cost-gated before launch. No canonical recipe change is
made before review.

## Prior evidence

The retained `B=32d` learning-rate law was fitted from 75 runs across four
widths and three training ratios in
`experiments/2026-08-04.01-dense-learning-rate-ratio`. It is not retuned here.

The earlier reduced-batch sweep in
`experiments/2026-08-03.02-dense-batch-cost-pilot` compared `B=16d` and `B=32d`
at d32 through d256. Its selected learning rates imply `16d / 32d` multipliers
from `0.67` to `0.74`, centered near `0.70`, but the multiplier was not directly
tuned at every width.

## Stage 1: learning-rate multiplier

Stage 1 holds each run to the samples represented by 8,000 `B=32d` steps. The
actual `B=16d` runs therefore train for exactly 16,000 steps. At d128 and d512,
the tested multipliers are `0.55`, `0.70`, and `0.85` times the canonical LR
prediction. A held-out d1024 run tests transfer of `0.70`.

All arms use seed 1, identical accepted-sample counts within a width, the
canonical loss and cooldown, and isolated W&B run names. The conservative cost
estimate assumes a `16d` step takes as long as the measured `32d` step:

| Width | Runs | Estimated hardware cost |
| ---: | ---: | ---: |
| d128 | 3 | `$0.14` |
| d512 | 3 | `$0.69` |
| d1024 | 1 | `$0.64` |
| **Total** | **7** | **`$1.47`** |

Launch and analysis details will be appended after completion.

### Initial results

All seven initial runs completed with their exact planned sample counts. Final
task-loss EMA favored the high `0.85` edge at both tuned widths:

| Width | `0.55x` | `0.70x` | `0.85x` |
| ---: | ---: | ---: | ---: |
| d128 | 3.3835 | 3.3824 | **3.3687** |
| d512 | 3.1075 | 3.0976 | **3.0912** |
| d1024 | n/a | 2.9894 | n/a |

The runs consumed about `$1.03` from recorded W&B runtimes and repository
hardware rates, below the conservative `$1.47` estimate. Because both tuned
widths selected the same high boundary, the first edge extension tests `1.00x`
at d128 and d512, plus `0.85x` and `1.00x` at the held-out d1024 width. Its
conservative incremental estimate is `$1.553`.

The first extension bracketed d128 at `0.85x`, but d512 improved slightly
through `1.00x` and d1024 improved materially through `1.00x`. Recorded spend
through both stages was about `$1.99`. The final common extension tested
`1.15x` at all three widths and a d1024-only `1.30x` boundary.

The d128 optimum is bracketed at `0.85x`. D512's `1.15x` arm improved slightly
on EMA but recorded one loss spike, disqualifying it under the learning-rate
protocol; `1.00x` is the retained stable point. D1024 continued to improve
through the stable `1.30x` boundary. A final d1024-only extension tests `1.50x`
and `1.70x`, with a conservative `$1.274` estimate.

Both final d1024 arms were disqualified: `1.50x` recorded two loss spikes and
`1.70x` recorded one. The stable LR points retained for the horizon ladder are
d128 `0.85x`, d512 `1.00x`, and d1024 `1.30x` the canonical `32d` LR law.

After implementation review, a follow-up d1280 calibration used 24,000 `16d`
steps, equivalent to 12,000 `32d` steps and above the fitted d1280 minimum. The
matched `1.15x` and `1.30x` arms had final EMA losses of `2.8768` and `2.8680`.
The `1.15x` arm recorded one spike; `1.30x` recorded none and is selected. The
pair cost `$2.218` from recorded runtimes, below its conservative `$3.978`
launch estimate.

That follow-up ran from a stale rounded-batch worktree: its batch was 24,576
(`19.2d`), not the canonical exact `16d` batch of 20,480. The `1.30x` value is
therefore provisional for exact d1280 `16d` and requires a matched confirmation.

## Stage 2: minimum-step ladder

At d128, d512, and d1024, matched `16d` and `32d` arms use samples equivalent
to 1,000, 2,000, and 4,000 `32d` steps. The `16d` arms therefore run for 2,000,
4,000, and 8,000 optimizer steps. An 8,000-step `32d` arm completes each ladder
and reuses the selected 16,000-step `16d` LR run above.

The initial conservative estimates were `$0.087` for d128 and `$0.417` for
d512. D128 first cleared the canonical loss/FLOPs trend at 4,000 optimizer
steps. D512 remained below trend at 8,000 steps and cleared it with the existing
16,000-step `16d` anchor. The ladder was therefore adapted before d1024:

- add a d512 `32d` validation at 16,000 steps;
- skip the clearly inadequate d1024 1,000/2,000-step pairs;
- run the d1024 4,000-step pair, 8,000-step `32d` arm, and 16,000-step `32d`
  validation, reusing the existing 16,000-step `16d` anchor.

The revised conservative estimates are `$0.230` for the d512 validation and
`$1.433` for d1024. Recorded spend, rather than the deliberately pessimistic
same-step-time envelope, remains the authoritative `$8` cost gate.

D1024 showed that a common optimizer-step threshold is insufficient: the
selected `16d` arm beat trend at 16,000 steps, while `32d` remained below trend
at 16,000 and is known to be healthy at the canonical 33,574-step horizon. The
experiment therefore adds d768 as an interpolation width, using a `1.15x`
interpolated `16d` LR multiplier and matched 8,000/16,000-step `32d` horizons.
Its four-run conservative estimate is `$1.640`; recorded spend before this
stage is `$5.952`.

## Results

Selection uses final `loss/task[ema=0.99]`. `EG_flops` is evaluated against the
current canonical dense loss/FLOPs curve. Values below `1.0x` indicate an
optimization deficit relative to expected scaling.

### Learning rate

| Width | Stable LR sweep (`16d / 32d` multiplier: loss) | Initial selection |
| ---: | --- | ---: |
| d128 | `0.55: 3.3835`, `0.70: 3.3824`, `0.85: 3.3687`, `1.00: 3.3838`, `1.15: 3.3900` | `0.85x` |
| d512 | `0.55: 3.1075`, `0.70: 3.0976`, `0.85: 3.0912`, `1.00: 3.0890` | `1.00x` |
| d1024 | `0.70: 2.9894`, `0.85: 2.9753`, `1.00: 2.9653`, `1.15: 2.9593`, `1.30: 2.9557` | `1.30x` |
| d1280 | `1.15: 2.8768` (1 spike), `1.30: 2.8680` | `1.30x` |

D512 `1.15x` had one spike. D1024 `1.50x` and `1.70x` had two and one
respectively. Because learning rate can plausibly cause instability, all three
are disqualified. The d768 horizon ladder used the interpolated stable `1.15x`
multiplier; its 16,000-step arm was spike-free.

Selected W&B runs: [d128](https://wandb.ai/maxence-frenette/chess-engine-4/runs/kgjibveh),
[d512](https://wandb.ai/maxence-frenette/chess-engine-4/runs/lq0gwr2b),
[d768](https://wandb.ai/maxence-frenette/chess-engine-4/runs/unhc161t),
[d1024](https://wandb.ai/maxence-frenette/chess-engine-4/runs/0bk74hfy), and
[d1280](https://wandb.ai/maxence-frenette/chess-engine-4/runs/mp297yl0).

The minimum-safe-horizon follow-up found spikes at d512, d768, and d1024 with
those initial selections. Rerunning at the next lower multiplier removed all
three spikes, so the recipe intentionally stores the conservative promoted
values in an explicit dictionary:

```text
{64: 0.85, 128: 0.85, 256: 0.85, 512: 0.85,
 768: 1.00, 1024: 1.15, 1280: 1.30}
```

The d64 and d256 entries preserve the prior implementation's clamped `0.85x`
choice; they are not presented as new direct tunes. The d512/d768/d1024 values
are the stable short-horizon retries, while d128 and d1280 were directly tuned.
No learning-rate scaling law is claimed.

### Optimizer steps

The `B16` and `B32` columns are matched at the same width and accepted samples.
The first number is optimizer steps; the following values are loss / `EG_flops`.

| Width | Equivalent `B32` horizon | `B32` | `B16` |
| ---: | ---: | ---: | ---: |
| d128 | 1,000 | 1k: 3.9481 / 0.305x | 2k: 3.8187 / 0.646x |
| d128 | 2,000 | 2k: 3.6954 / 0.709x | 4k: 3.6501 / 0.966x |
| d128 | 4,000 | 4k: 3.5000 / 1.476x | 8k: 3.4923 / 1.570x |
| d128 | 8,000 | 8k: 3.3865 / 1.926x | 16k: 3.3687 / 2.263x |
| d512 | 1,000 | 1k: 3.7917 / 0.018x | 2k: 3.6367 / 0.050x |
| d512 | 2,000 | 2k: 3.4249 / 0.128x | 4k: 3.3838 / 0.184x |
| d512 | 4,000 | 4k: 3.2304 / 0.418x | 8k: 3.2161 / 0.489x |
| d512 | 8,000 | 8k: 3.1024 / 0.956x | 16k: 3.0890 / 1.141x |
| d512 | 16,000 | 16k: 3.0120 / 1.707x | n/a |
| d768 | 8,000 | 8k: 3.0445 / 0.662x | 16k: 3.0134 / 1.049x |
| d768 | 16,000 | 16k: 2.9640 / 1.154x | 32k: 2.9235 / 2.343x (1 spike) |
| d1024 | 4,000 | 4k: 3.1019 / 0.262x | 8k: 3.0706 / 0.398x (5 spikes) |
| d1024 | 8,000 | 8k: 3.0012 / 0.550x | 16k: 2.9557 / 1.154x |
| d1024 | 16,000 | 16k: 2.9323 / 0.869x | n/a |

D1024's canonical 33,574-step `32d` run is the healthy upper bracket. This
establishes that batch choice is not governed by optimizer-step count alone:
at d1024, `16d` is healthy at 16,000 steps while `32d` is not.

## Proposed conservative recipe

The step boundary is represented by a smooth monotone power law fitted to the
`32d` crossings. The same curve controls both batch selection and rejection:

```text
critical_steps(d_model) = 62.7575303963433 * d_model^0.8073049254601639

if steps_32d >= critical_steps: use 32d
else if 2 * steps_32d >= critical_steps: use 16d
else: reject
```

There is no integer ceiling in the fitted law. The recipe first computes the
canonical `32d` integer step count. Switching to `16d` halves the batch exactly
and doubles that step count exactly, preserving accepted samples.

As a sensitivity check, the same four `32d` crossings were fitted in log space
against width, total parameters, and FLOPs per sample. FLOPs per sample fits
slightly best, with total parameters effectively tied:

| Scale coordinate | In-sample residual factor | Leave-one-width-out factor |
| --- | ---: | ---: |
| `d_model` | `1.049x` | `1.214x` |
| total parameters | `1.035x` | `1.141x` |
| FLOPs per sample | `1.035x` | `1.138x` |

The differences are small and there are only four crossings. For this fixed
depth-8 dense family, all three coordinates are nearly deterministic transforms
of one another. FLOPs per sample means model compute per accepted sample here;
total training compute would include the unknown step boundary and would make
the fit circular. The reviewed implementation retains the simpler width fit.

## Cost and verdict

The original forty runs completed for `$7.404` using recorded W&B runtimes and
repository hardware rates, below their authorized `$8` ceiling. The separately
requested two-run d1280 extension cost `$2.218`, for `$9.623` cumulatively. All
commands are generated by `launch_lr.py` and `launch_steps.py`; every run uses
the repository W&B project and an isolated `dense-minsteps-*` name.

The experiment supports an adaptive batch rule and hard rejection below the
validated minimum. After review, the canonical dense recipe uses the shared
smooth `32d`-derived boundary above and the explicit `16d` learning-rate
dictionary.

## Methodological source

The cross-scale ladder follows the methodology of *MAI-Thinking-1*. The paper
supports testing recipe changes across scales; the LR multiplier and minimum
steps are determined only from this project's matched experiments.
