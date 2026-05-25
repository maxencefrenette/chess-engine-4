# MLP-MoE16a2 Aux, Batch, LR Sweep

This sweep tuned `router_aux`, then jointly tuned batch size and learning rate for `1e14` and `1e15`, then transferred the learning-rate and aux-loss findings to `1e16`. Architecture stayed fixed within each budget, and the MoE family constraints were unchanged: 16 experts, 2 active experts, `expert_mlp_ratio = 2.0`.

Metrics are post-hoc W&B tail means over the last 100 rows where both `loss/total` and `metrics/policy_top1` are present.

## Winners

| Budget | Previous Loss | New Loss | Winning Run | Batch | LR | Aux | W&B |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `1e14` | 3.9048 | 3.8847 | `moe16a2-grid2-1e14-b384-lr2e-3-a3e-3` | 384 | 2e-3 | 0.003 | https://wandb.ai/maxence-frenette/chess-engine-4/runs/chhj4xnd |
| `1e15` | 3.5940 | 3.5830 | `moe16a2-grid2-1e15-b1024-lr8.5e-4-a1e-3` | 1,024 | 8.5e-4 | 0.001 | https://wandb.ai/maxence-frenette/chess-engine-4/runs/wtp9t8mt |
| `1e16` | 3.3726 | 3.3586 | `moe16a2-1e16-b1024-lr4e-4-a1e-3` | 1,024 | 4e-4 | 0.001 | https://wandb.ai/maxence-frenette/chess-engine-4/runs/yatiw0yl |

## Aux Loss

- `1e14` slightly preferred `router_aux = 0.003` over `0.001` and `0.01`; `0.03` was worse.
- `1e15` preferred `router_aux = 0.001`; increasing it monotonically worsened tail loss in this sweep.
- The transfer to `1e16` used `router_aux = 0.001`, which beat the previous `0.01` baseline when paired with a higher LR.

## Batch And LR

- `1e14` benefited from doubling batch size from 192 to 384 when LR was raised. The best tail loss was at `b384/lr2e-3`. Larger `b512` was worse.
- `1e15` stayed best at `b1024`, but LR needed to be higher than the previous `5e-4`. The best tail loss was `b1024/lr8.5e-4`, with `lr1e-3` nearly tied but slightly worse.
- Larger `1e15` batches (`1536`, `2048`) did not win in tail loss, even when LR was raised.

## Baseline Updates

The `configs/mlp_moe16a2/*.toml` files and `experiments/best-runs-mlp_moe16a2.toml` now point to the three winning rows above. The local ignored scaling-law report was regenerated from the updated best-runs file.
