# MLP-MoE 16a2 1e21 Baseline

## Goal

Train a `1e21` compute-budget baseline for the `mlp_moe16a2` model family and
save the final checkpoint for later lc0 ONNX evaluation.

This run tests whether the MoE family can beat the dense MLP at the same
step-adjusted compute budget while keeping all experts alive.

## Run

```bash
uv run train-modal --config configs/mlp_moe16a2/1e20.toml --compute-budget 1e21 --d-model 640 --depth 5 --batch-size 24576 --lr 3e-4 --router-aux 0.01 --save-checkpoints --wandb-name moe16a2-1e21-d640x5-b24576-lr3e-4-aux1e-2-finalckpt
```

W&B:

```text
https://wandb.ai/maxence-frenette/chess-engine-4/runs/9ve4eoad
```

Checkpoint:

```text
/artifacts/checkpoints/moe16a2-1e21-d640x5-b24576-lr3e-4-aux1e-2-finalckpt-final.pt
```

## Configuration

| Setting | Value |
| --- | ---: |
| Model | `mlp_moe16a2` |
| Shape | `d640x5` |
| Experts | `16` |
| Active experts | `2` |
| Expert MLP ratio | `2.0` |
| Parameters | `202.4M` |
| Batch size | `24576` |
| LR | `3e-4` |
| Router aux weight | `0.01` |
| Max grad norm | `1.0` |
| LR cooldown | `0.1` |
| GPU | `l4` |

## Results

| Metric | Value |
| --- | ---: |
| Steps | `14,834` |
| Samples | `364,560,384` |
| Compute seen | `1.000e21` |
| Final task loss | `2.9939` |
| `loss/task[ema=0.99]` | `3.0092` |
| `loss_upper_1sd` | `3.0429` |
| Policy top-1 EMA | `0.4671` |
| Router loss | `1.0385` |
| Dead experts | `0` |
| Max dead experts | `0` |
| Runtime | `5275.8s` |

## Dense Comparison

| Model | Shape | Loss EMA | Loss upper 1 SD | Policy top-1 EMA |
| --- | --- | ---: | ---: | ---: |
| dense MLP `1e21` | `d768x5` | `3.0394` | `3.0762` | `0.4513` |
| MoE `1e21` | `d640x5` | `3.0092` | `3.0429` | `0.4671` |

The MoE is slightly better at the same step-adjusted compute budget:

- `-0.0302` loss EMA
- `-0.0333` loss upper 1 SD
- `+1.58` policy top-1 percentage points

## Notes

Lowering router aux from `0.03` to `0.01` worked at this scale. The run ended
with zero dead experts and router loss near the balanced target.

There was a small loss spike during training, so this run may not be the best
this configuration can do. The final metrics are still a clean win over the
dense `1e21` baseline, but the spike means the result should not be treated as
the fully optimized MoE point.
