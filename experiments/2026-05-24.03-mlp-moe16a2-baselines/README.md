# MLP-MoE16a2 Baselines

Baseline runs for the `mlp_moe16a2` family after fixing MoE FLOPs accounting. This family uses 16 routed experts, activates 2 experts per sample, and uses `expert_mlp_ratio = 2.0`.

Metrics below use the standard post-hoc W&B tail methodology: the mean of the last 100 rows where both `loss/total` and `metrics/policy_top1` are present.

| Budget | Run | Model | Batch | LR | Samples | Params | Loss | Policy Top-1 | Runtime | W&B |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `1e14` | `mlp-moe16a2-baseline-1e14` | `d32x1` | 192 | 1e-3 | 6,416,064 | 389,670 | 3.9048 | 0.2328 | 6.9m | https://wandb.ai/maxence-frenette/chess-engine-4/runs/bky4e3br |
| `1e15` | `mlp-moe16a2-baseline-1e15` | `d80x3` | 512 | 7.071e-4 | 19,941,376 | 2,571,382 | 3.6246 | 0.3077 | 17.4m | https://wandb.ai/maxence-frenette/chess-engine-4/runs/zzwly9ow |
| `1e16` | `mlp-moe16a2-baseline-1e16` | `d160x6` | 1,024 | 3e-4 | 55,296,000 | 16,207,782 | 3.3726 | 0.3678 | 43.4m | https://wandb.ai/maxence-frenette/chess-engine-4/runs/uptnni2r |

The `1e16` run lands very close to the dense MLP baseline loss while using substantially more total parameters due to inactive experts. The `1e14` run looks underwhelming by loss, and the family probably needs shape tuning rather than directly inheriting the dense MLP shapes.

The scaling-law inputs for this family are stored in `experiments/best-runs-mlp_moe16a2.toml`. The generated report is under `reports/scaling-laws/mlp_moe16a2/1e17/`.
