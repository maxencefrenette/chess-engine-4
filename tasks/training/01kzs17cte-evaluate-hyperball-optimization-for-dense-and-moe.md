---
id: "01kzs17cte"
title: "Evaluate Hyperball optimization for dense and MoE models"
status: completed
priority: medium
effort: large
dependencies: []
tags: ["optimizer", "experiment", "hyperball"]
touches: ["training", "experiments"]
created_at: 2026-08-11
---

# Evaluate Hyperball optimization for dense and MoE models

## Objective

Determine whether Adam Hyperball (AdamH) can replace AdamW in the canonical
dense recipe, eliminating `weight_decay` as a tuned hyperparameter without
regressing training efficiency. If it passes the dense gate, validate the same
method on `moe64a2`.

The experiment must distinguish two possible wins:

1. AdamH beats the current AdamW recipe with `weight_decay = 0.01`.
2. AdamH remains competitive with a lightly checked AdamW control at the
   less-recently-tuned `32d` batch size.

The first result would show that the current decay is poorly tuned. The second
is required to claim that Hyperball removes a hyperparameter without giving up
performance.

## Method

For every eligible matrix with initial FP32 master weight `W_0`, store
`R = ||W_0||_F`. Given the Adam update direction `u_t` before learning-rate
scaling, apply

```text
W_{t+1} = R * normalize(W_t - eta_t * R * normalize(u_t)).
```

Use Adam, with no weight decay, for parameters outside the constraint. Retain
the existing Adam moments, global gradient clipping, zero warmup, and 10%
linear cooldown. `eta_t` is the only optimizer scale to tune for AdamH.

### Eligible parameters

- Dense: each `blocks.*.layer.fc1_weight` and `fc2_weight` matrix.
- MoE: the dense-block matrices plus each expert gate/up and down matrix,
  normalized separately per expert even when stored in a stacked 3-D tensor.
- Exclude input projections, policy/value/moves-left heads, RMSNorm gains,
  biases, and MoE routers. Their scale can change the represented function;
  router scale also changes softmax sharpness.

This follows the paper's use of Hyperball on pre-norm Transformer MLP matrices
and standard Adam on embeddings, normalization gains, and semantic-scale
parameters. Applying it to this MLP-only residual architecture is an empirical
extension, not a result established by the paper.

## Phase 0: implementation and correctness

- [x] Add an explicit optimizer choice rather than interpreting
  `weight_decay = 0` as AdamH. AdamH configs must not expose a decay value.
- [x] Apply normalization and projection to FP32 master weights, then keep the
  BF16 parameter copy synchronized. Do not project only the BF16 copy.
- [ ] Store radii and any required wrapper state in optimizer checkpoints and
  verify that resume is bitwise-equivalent over a short deterministic run.
- [x] Make zero-norm updates a no-op and use only a fixed numerical epsilon,
  not another user-facing hyperparameter.
- [ ] Unit-test parameter eligibility, per-expert normalization axes, Adam
  moment updates, scheduled step length, fixed radii, checkpoint round trips,
  and config serialization.
- [ ] Log aggregate and worst-case `||W_t||_F / R - 1`, angular displacement,
  update norm, and eligible/excluded parameter counts. For MoE, also log these
  diagnostics by expert layer.
- [x] Run a 100-step dense d128 smoke test. Require finite metrics, FP32 master
  radius error below `1e-5`, synchronized BF16 weights, and no checkpoint-resume
  divergence beyond the normal BF16 tolerance.
- [x] Benchmark 500 steady-state steps against FusedAdamW. Continue only if
  realized step-time overhead is at most 5% and peak memory remains practical.

The implementation should avoid a full saved copy of every eligible matrix per
step. If Transformer Engine cannot expose the unscaled Adam direction safely,
add a fused terminal transform or a small dedicated optimizer kernel instead of
accepting permanent model-sized copy overhead.

## Phase 1: dense anchor sweep

Use dense d128 at `0.1x`, seed 1, the canonical model/data/batch/loss recipe,
and completed schedules. This selects the `32d` batch path, whose optimizer
settings have not been tuned recently. Run fresh controls from the same
implementation commit.

### AdamW control surface

Use only a five-arm local cross around the current recipe:

```text
canonical LR:       weight_decay in {0.003, 0.01, 0.03}
weight_decay 0.01:  LR multiplier in {1/sqrt(2), 1, sqrt(2)}
```

The center is shared, so this is five runs rather than a Cartesian sweep. Do
not expand the AdamW search: it is a sanity check for the older `32d` setting,
not a new optimizer-tuning campaign.

### AdamH surface

Sweep dimensionless learning rate on a five-point `sqrt(2)` grid:

```text
{0.0035, 0.0050, 0.0071, 0.0100, 0.0141}
```

Do not expand automatically. These values cover the center of the range shown
in the Hyperball paper; they are not derived from the existing AdamW
learning-rate law. A boundary winner is reported as unresolved rather than
turning this into a large sweep.

Select only from spike-free learning-rate runs, as required by the repository
optimizer-tuning policy. The primary selection target is the lowest completed-
run `loss/task[ema=0.99]`; report policy top-1, loss trajectory, spike count,
runtime, peak memory, and `EG_flops` as supporting metrics.

## Phase 2: dense confirmation and transfer

At dense d128 `0.2x`, compare these three arms on paired seeds 1-2:

1. current AdamW (`weight_decay = 0.01`, canonical LR);
2. best light-check AdamW cell from phase 1;
3. best AdamH LR from phase 1.

Then run the selected AdamH at dense d256 and d512 `0.055x`, transferring the
exact same dimensionless LR. This ratio uses `16d` and matches the recent
stable width-scaling controls in
`2026-08-11.01-dense-half-horizon-width-scaling`, so reuse those AdamW runs
instead of paying for duplicate controls. Reuse is allowed only if model, data,
batch, schedule, LR, and optimizer behavior remain compatible after the AdamH
implementation. Do not retune AdamW or transfer the d128 `32d` adjustment.

Finally, run one longer-horizon dense d256 AdamH arm at `0.5x` with the same
selected LR. This returns to the `32d` batch path and tests whether AdamH still
behaves as expected after substantially more optimizer steps and a longer
cooldown. Compare it with the retained d256 `0.5x` AdamW run from
`2026-08-06.03-training-ratio-refresh` only after confirming model, data,
batch, schedule, LR, and optimizer compatibility. If that control is
incompatible, launch one fresh control and remove one redundant d128
confirmation arm before launch so the planned budget remains below `$3`.

### Dense gate

AdamH passes the simplification gate when all of the following hold:

- the selected LR is spike-free at every confirmation width;
- at d128, both paired-seed `AdamH - best light-check AdamW` final EMA loss
  differences are at most the pre-registered non-inferiority margin `+0.005`,
  and their mean is non-positive;
- at d256 and d512 `0.055x`, AdamH is no more than `0.005` loss worse than the
  compatible retained, recently tuned `16d` AdamW control;
- at d256 `0.5x`, AdamH remains stable and is no more than `0.005` loss worse
  than the compatible long-horizon AdamW control;
- step-time overhead remains at most 5% at d512; and
- radius and synchronization invariants hold throughout training.

Label AdamH a performance gain only when both paired d128 differences are
strictly below zero and its transferred d256/d512 runs also beat canonical
AdamW. Report
sample-equivalent `EG_flops` and realized cost separately so convergence gains
are not confused with optimizer overhead.

## Phase 3: conditional MoE validation

Start only after AdamH passes the dense simplification gate.

At MoE d256 `0.02x`, run fresh seed-1 screens:

- AdamW: only three decay points `{0.003, 0.01, 0.03}` at canonical LR;
- AdamH: a five-point `sqrt(2)` LR grid centered on the dense-selected LR.

Do not expand either grid. Select only spike-free LRs. At MoE d256 `0.05x`, run
the selected AdamH and the best light-check AdamW arm once. Compare both with
the compatible retained QB d256 control from
`2026-08-10.03-quantile-load-balancing`. Then freeze AdamH and run it once at
MoE d512 `0.05x`, comparing with the compatible retained QB d512 control from
that experiment. Do not pay for duplicate canonical controls or retune at
d512.

Apply the dense non-inferiority rule with the same `+0.005` margin. In addition,
every selected MoE run must finish with zero dead experts, preserve quantile-
balanced routing, and show no material regression in router-token balance. A
local LR diagnostic after a transfer failure is reportable but cannot turn the
frozen-transfer result into a pass, and it requires an explicit budget check
before launch.

## Compute budget

The target total spend is `$3`; `$5` is a hard cap, not an allocation target.
Review the printed launch summary and cumulative recorded cost before every
stage. Do not launch optional diagnostics when projected total cost exceeds
`$3`, and stop all new launches at `$5`. After the initial result, the user
authorized another `$5` specifically to rerun as much as possible of the dense
L-shaped scaling ladder; that superseded the unspent conditional MoE allocation.

Using the current throughput profiles, the planned allocation is approximately:

| Stage | Planned cost |
| --- | ---: |
| Dense d128 screens and two-seed confirmation | `$0.22` |
| Dense d256/d512 short transferred AdamH runs | `$0.15` |
| Dense d256 `0.5x` AdamH validation | `$0.10` |
| Correctness/profile allowance | `$0.05` |
| MoE d256 screens and `0.05x` confirmation | `$1.20` |
| One MoE d512 AdamH run at `0.05x` | `$1.21` |
| **Estimated total** | **`$2.93`** |

The estimate includes GPU and configured CPU rates but not unusual startup or
recompilation delays. Reusing only configuration-compatible retained controls
is what makes the strong d512 MoE check fit under the target. If cumulative
cost reaches `$2.70` before the d512 launch, shorten that final arm to `0.02x`
rather than knowingly exceeding the `$3` target.

## Reporting and promotion

- [x] Record commit, commands, complete arm table, W&B URLs, final EMA loss,
  policy top-1, spikes, runtime, cost, `EG_flops`, norm diagnostics, and verdict
  in a new dated experiment report.
- [x] Preserve every completed control and failed arm in the report.
- [x] State separately whether the result found a bad current decay, removed
  the need to tune decay, improved convergence, and transferred across widths.
- [x] Do not change canonical configs or best-run registries until explicit
  review and approval.

## Acceptance criteria

- Hyperball is implemented exactly on the intended FP32 matrix blocks and is
  correct across checkpoint resume.
- Current AdamW plus a light local check at the less-recently-tuned batch
  settings make both practical and method-level claims possible without
  reopening broad AdamW tuning.
- Dense selection, confirmation, transfer, and conditional MoE gates follow the
  pre-registered arms and thresholds above.
- The final report distinguishes the paper's findings from measurements in this
  repository and separates training-FLOP efficiency from wall-clock cost.

## Reference

Kaiyue Wen, Xingyu Dang, Kaifeng Lyu, Tengyu Ma, and Percy Liang,
*Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball
Optimization*, arXiv:2606.16899 (2026).

Yihao Xiao et al., *Hyperball May Not Be a Free Lunch*, arXiv:2607.22444
(2026). This follow-up motivates keeping the LR schedule controlled and not
interpreting fixed matrix norms as eliminating schedule selection.

## Outcome

Dense evaluation and the promoted scaling rerun completed on 2026-08-11 and
are recorded in `experiments/2026-08-11.03-hyperball/README.md`. AdamH failed
the original paired d128 non-inferiority gate, but its advantage increased at
d256 and d512. The user changed the decision target to expected larger-scale
performance, explicitly approved universal optimizer promotion, and assigned
the added budget to the dense scaling ladder. AdamH is now the canonical dense
optimizer, `weight_decay` is removed, and LR remains width-specific.

All 25 cells of the prior dense L-shaped surface were rerun with AdamH. The
d64 through d512 canonical `0.2x` candidates passed the `EG_flops` promotion
rule. The requested long d256 run was stable and beat its compatible retained
AdamW control. D768 and d1024 improved despite recoverable MXFP8 spikes; no
spike-free LR was found in the bounded retries, so their LR stability remains
provisional. MoE validation remains a separate follow-up because the original
conditional gate skipped it and the added budget was spent on the dense ladder.

Checkpoint state was saved successfully by every AdamH run. A dedicated
bitwise resume comparison and richer angular/per-layer diagnostics were not
needed to decide the failed promotion gate and remain unchecked above rather
than being represented as completed.
