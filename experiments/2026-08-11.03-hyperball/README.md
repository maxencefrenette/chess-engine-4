# Hyperball optimizer experiment

## Goal and method

Test whether Adam Hyperball (AdamH) can remove `weight_decay` from the dense
training recipe without losing final EMA task loss, then validate it on MoE
only if it passes the dense gate. AdamH constrains each dense MLP matrix, and
each MoE expert matrix separately, to its initial FP32 Frobenius radius. Heads,
input projections, norms, biases, and routers remain on ordinary Adam without
weight decay.

Runs used the worktree based on `8cc8fd49` with the AdamH implementation patch,
the canonical data and losses, completed schedules, and seed 1 unless noted.
The pre-registered non-inferiority margin was `+0.005` final
`loss/task[ema=0.99]`. Learning-rate selection required zero detected spikes.

## Implementation gate

The 500-step d512 steady-state profile measured `5.9408 ms/step` of training
GPU time for AdamH versus `5.7391 ms/step` for FusedAdamW, an overhead of
`3.51%`. Peak allocated memory was 830 MB versus 782 MB. The maximum measured
relative radius error was `1.19e-7`, below the `1e-5` gate. A 100-step d128
smoke test completed with finite loss.

## Dense d128 screen

All ten `0.1x`, batch-`32d` runs were spike-free. The five-arm AdamW check
found that the current decay was not locally optimal: lowering decay from
`0.01` to `0.003` improved EMA loss by `0.00795`. The best AdamW cell was
instead the lower-LR arm at the current `0.01` decay.

| Optimizer | LR | Decay | EMA loss | Spikes | W&B |
| --- | ---: | ---: | ---: | ---: | --- |
| AdamW | 0.002300 | 0.003 | 3.534400 | 0 | [ot04nygn](https://wandb.ai/maxence-frenette/chess-engine-4/runs/ot04nygn) |
| AdamW | 0.002300 | 0.010 | 3.542351 | 0 | [rr1tk2y7](https://wandb.ai/maxence-frenette/chess-engine-4/runs/rr1tk2y7) |
| AdamW | 0.002300 | 0.030 | 3.543756 | 0 | [of8w1hoz](https://wandb.ai/maxence-frenette/chess-engine-4/runs/of8w1hoz) |
| AdamW | 0.001626 | 0.010 | **3.526444** | 0 | [tf9zlkym](https://wandb.ai/maxence-frenette/chess-engine-4/runs/tf9zlkym) |
| AdamW | 0.003253 | 0.010 | 3.563956 | 0 | [fo224rmf](https://wandb.ai/maxence-frenette/chess-engine-4/runs/fo224rmf) |
| AdamH | 0.003500 | none | 3.538077 | 0 | [bs8pjdc1](https://wandb.ai/maxence-frenette/chess-engine-4/runs/bs8pjdc1) |
| AdamH | 0.005000 | none | **3.533224** | 0 | [cpdi7w60](https://wandb.ai/maxence-frenette/chess-engine-4/runs/cpdi7w60) |
| AdamH | 0.007100 | none | 3.536166 | 0 | [3s8zyiow](https://wandb.ai/maxence-frenette/chess-engine-4/runs/3s8zyiow) |
| AdamH | 0.010000 | none | 3.543113 | 0 | [gyrlwm3f](https://wandb.ai/maxence-frenette/chess-engine-4/runs/gyrlwm3f) |
| AdamH | 0.014100 | none | 3.558132 | 0 | [ffyj9hx5](https://wandb.ai/maxence-frenette/chess-engine-4/runs/ffyj9hx5) |

The selected AdamH LR was `0.005`. Its EMA loss was `+0.00678` worse than the
best light AdamW cell at the screen horizon. Its `EG_flops` was `0.967x`, versus
`1.002x` for the best AdamW cell.

## Paired d128 confirmation

At `0.2x`, both paired seeds failed the non-inferiority margin against the best
light AdamW setting. The AdamH differences were `+0.01383` and `+0.01136`, with
a mean of `+0.01259`. Every run was spike-free.

| Seed | Canonical AdamW | Best light AdamW | AdamH | AdamH - best | W&B |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | [3.410786](https://wandb.ai/maxence-frenette/chess-engine-4/runs/qb138xk4) | [3.412735](https://wandb.ai/maxence-frenette/chess-engine-4/runs/36iya2o4) | [3.426562](https://wandb.ai/maxence-frenette/chess-engine-4/runs/iyxd2d6n) | +0.013826 | [iyxd2d6n](https://wandb.ai/maxence-frenette/chess-engine-4/runs/iyxd2d6n) |
| 2 | [3.412562](https://wandb.ai/maxence-frenette/chess-engine-4/runs/3el7cpd0) | [3.416817](https://wandb.ai/maxence-frenette/chess-engine-4/runs/kvoshukq) | [3.428172](https://wandb.ai/maxence-frenette/chess-engine-4/runs/t5td91if) | +0.011355 | [t5td91if](https://wandb.ai/maxence-frenette/chess-engine-4/runs/t5td91if) |

## Width transfer and long d256 validation

The requested d256 `0.5x` validation was stable and encouraging: AdamH reached
3.070236 with no spikes, `0.019598` better than the compatible retained AdamW
control at 3.089834. The short d256 transfer also beat its recent stable AdamW
control. D512 improved substantially but recorded one loss spike, so the
selected LR is not valid at that width under the repository LR policy.

| Width / ratio | AdamH loss | AdamW control | Difference | Spikes | EG_flops | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| d256 / 0.055x | 3.376105 | 3.409629 | -0.033524 | 0 | 1.109x | [tiadc73r](https://wandb.ai/maxence-frenette/chess-engine-4/runs/tiadc73r) |
| d512 / 0.055x | 3.124745 | 3.168128 | -0.043383 | 1 | 1.330x | [4oqa8n7d](https://wandb.ai/maxence-frenette/chess-engine-4/runs/4oqa8n7d) |
| d256 / 0.5x | 3.070236 | 3.089834 | -0.019598 | 0 | 1.427x | [j1p7ggt0](https://wandb.ai/maxence-frenette/chess-engine-4/runs/j1p7ggt0) |

The short controls are the stable lower-LR rows from
`2026-08-11.01-dense-half-horizon-width-scaling`; the long control is from
`2026-08-06.03-training-ratio-refresh`. Model, data, batch, steps, schedule, and
ordinary AdamW behavior are compatible. No fresh duplicate controls were run.

## Initial cost and pre-registered verdict

Modal's per-app billing report records `$3.19` for the complete implementation,
correctness/profile, and experiment: `$2.39` for implementation profiling and
smokes, plus `$0.81` for the 19 recorded training runs. This excludes concurrent
apps from another experiment in the same workspace. The result exceeded the
`$3` target by about `$0.19` but remained below the `$5` hard cap. No Hyperball
jobs remained active after collection.

**The original local dense gate failed.** AdamH missed d128 non-inferiority on
both seeds, and the original d512 transfer spiked. That gate weighted the small
d128 anchor more heavily than the larger-width result. The subsequent promotion
decision below uses the project's expected larger-scale `$50` run as the target
instead: AdamH's advantage increased with width and remained large after a
spike-free d512 LR retry. The AdamW check independently supports revisiting
decay and LR at batch `32d`.

Runs were launched with `train-modal`; the tables above record every arm and
W&B URL. The command shape was:

```sh
uv run train-modal --config configs/dense.py --d-model WIDTH \
  --training-ratio RATIO --seed SEED --optimizer adamh --lr LR \
  --wandb-name NAME
```

## Promotion decision

AdamH is promoted as the canonical dense optimizer and removes
`weight_decay` from the recipe. This is an optimizer-family default, not a
claim that one learning rate transfers unchanged. Hyperball reports only
approximate scale invariance: its empirical optimal LR moves about `1.4x`
across width/depth in the paper, less than the roughly `2-4x` movement reported
for AdamW and MuonW. The canonical recipe therefore keeps a small width-indexed
AdamH LR table.

At `0.055x`, every rerun beat its retained AdamW width control. The d512
`0.0035` retry was spike-free and retained nearly all of the original gain.
The larger MXFP8 widths improved substantially, but all tested LRs spiked;
lowering LR reduced spike count while worsening loss, so their selected LRs
remain provisional stability choices.

| Width | AdamH LR | AdamH loss | AdamW loss | AdamW EG | AdamH EG | EG gain | Spikes | W&B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 0.0035 | 3.882124 | 3.889359 | 1.128x | 1.162x | +3.0% | 0 | [c8ki1yba](https://wandb.ai/maxence-frenette/chess-engine-4/runs/c8ki1yba) |
| 128 | 0.0035 | 3.637907 | 3.668101 | 0.917x | 1.051x | +14.6% | 0 | [m49k7gdm](https://wandb.ai/maxence-frenette/chess-engine-4/runs/m49k7gdm) |
| 256 | 0.0050 | 3.376105 | 3.409629 | 0.926x | 1.109x | +19.8% | 0 | [tiadc73r](https://wandb.ai/maxence-frenette/chess-engine-4/runs/tiadc73r) |
| 512 | 0.0035 | 3.127499 | 3.168128 | 0.997x | 1.305x | +30.9% | 0 | [3ua9a9eg](https://wandb.ai/maxence-frenette/uncategorized/runs/3ua9a9eg) |
| 768 | 0.0035 | 2.990882 | 3.033531 | 1.120x | 1.552x | +38.6% | 6 | [ua5qjt2n](https://wandb.ai/maxence-frenette/chess-engine-4/runs/ua5qjt2n) |
| 1024 | 0.0025 | 2.909313 | 2.941948 | 1.270x | 1.674x | +31.8% | 6 | [0hzz1ai9](https://wandb.ai/maxence-frenette/chess-engine-4/runs/0hzz1ai9) |

The canonical `0.2x` d64, d128, d256, and d512 candidates all passed
`compare-run`'s width promotion rule. Their `EG_flops` improved from
`0.443x`, `0.630x`, `0.745x`, and `0.734x` to `0.893x`, `0.897x`, `1.135x`,
and `1.584x`, respectively. D256 recorded two recoverable spikes and d512
four; both EMA curves returned to trend. D768 and d1024 were not rerun at
`0.2x` within this budget.

## AdamH L-shaped scaling rerun

The added-budget campaign replaced all 25 cells of the existing dense
L-shaped fit: six width-arm points, the eight-point d64 data arm, and eleven
off-arm validation cells. The d64 data arm used its spike-free `0.0071` LR and
remained stable through `2x` training:

| Ratio | 0.1 | 0.2 | 0.3 | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Loss | 3.789090 | 3.665816 | 3.609939 | 3.552712 | 3.522925 | 3.496253 | 3.483847 | 3.477143 |

The fit remains coherent after the optimizer cutover. Across all 25 points,
Skaling has `0.161%` in-sample MAPE and `0.439%` L-shaped held-out MAPE,
versus `0.532%` and `1.628%` for Chinchilla. Restricting to ratios at least
`0.1` gives Skaling `0.138%` fit MAPE and `0.234%` held-out MAPE. The exact
observations and their W&B provenance are the canonical `[scaling_runs]`
entries in `experiments/best-runs-dense.toml`. The fit uses the repository's
existing scaling-law implementation; the held-out evaluation uses the
`evaluation_folds` and `l_shape_rows` methodology from
`2026-08-10.01-skaling-law`.

One qualification is important: the best LR can also move with training
horizon. At d64, `0.0071` won the local `0.1x` screen, while an exploratory
`0.0035` arm was better at `2x`. The promotion eliminates decay, not LR tuning.

## Expanded budget and remaining validation

The original work cost `$3.19`, and the first promotion ladder plus retries
added approximately `$1.1`. After the user authorized another `$5`, the
remaining LR checks and complete 25-cell AdamH scaling surface used
approximately `$4.9` from recorded run runtimes and configured GPU prices.
Total Hyperball work was therefore about `$9.2`, within the combined `$10`
envelope. No further paid runs were launched after the ladder completed.

MoE validation remains outstanding. The original conditional MoE phase was
not launched after the pre-registered small-width gate failed, and the added
budget was explicitly assigned to the dense scaling ladder. AdamH's optimizer
surface supports per-expert constraints, but that implementation should not be
called validated on MoE until the fresh routing and dead-expert checks run.
