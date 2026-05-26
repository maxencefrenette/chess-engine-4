# MLP k=2 baselines

This is a full MLP baseline cutover to the same step-adjusted compute convention as the MoE family. The MLP configs now use `step_penalty_k = 2.0`, and the old `1e14`/`1e15`/`1e16` MLP configs and best-run entries were removed.

To save compute, these runs reuse the current MoE baseline hyperparameters at the matching budget labels, but use the dense MLP family with `mlp_ratio = 4.0`.

| Budget | Shape | Batch | LR | Steps | Params | Loss | Loss + 1 SD | Policy top-1 | W&B |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `1e18` | `d80x3` | 2048 | 0.0012 | 8999 | 954,742 | 3.6548 | 3.7856 | 0.2896 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/uc82h12d) |
| `1e19` | `d192x4` | 4096 | 0.0004 | 10592 | 3,505,286 | 3.4514 | 3.5252 | 0.3370 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/540uftct) |

## Notes

- The new `experiments/best-runs-mlp.toml` only includes these `1e18` and `1e19` runs.
- These are baseline replacements, not tuned dense MLP runs. The shapes and optimizer settings were copied from the current MoE baselines for a cheap same-budget comparison.
- The `1e19` baseline improves the smoothed loss over `1e18`, with lower loss variance and higher policy top-1.
