# Root-Target Baselines

## Goal

Reset the baseline runs after switching the value and moves-left training targets from the result row to the root search row:

- value target: `root_q/root_d`
- moves-left target: `root_m`

Because the loss definition changed, these runs replace the previous `experiments/best-runs-mlp.toml` entries even though the model shapes and compute budgets are unchanged.

## Results

| Budget | Shape | Checkpoint | W&B | Tail loss | Policy top-1 |
| --- | --- | --- | --- | ---: | ---: |
| `1e14` | `d32x1` | `/artifacts/checkpoints/baseline-root-ckpt-1e14-final.pt` | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/ndwnfs7n) | `3.8842` | `0.2360` |
| `1e15` | `d80x3` | `/artifacts/checkpoints/baseline-root-ckpt-1e15-final.pt` | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/8t0jrso3) | `3.6072` | `0.3024` |
| `1e16` | `d160x6` | `/artifacts/checkpoints/baseline-root-ckpt-1e16-final.pt` | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/con1n4ql) | `3.3754` | `0.3558` |

`Tail loss` and `policy_top1` are means over the last 100 W&B rows where both metrics are present.

## Notes

The total loss is not directly comparable to older result-target baselines because the value and moves-left targets changed. Policy top-1 remains comparable, and it is slightly higher at all three budgets.

All three runs saved final checkpoints in the Modal artifact volume. `experiments/best-runs-mlp.toml` now points to these root-target runs.
