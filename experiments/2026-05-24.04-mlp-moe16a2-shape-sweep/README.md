# MLP-MoE16a2 Shape Sweep

Goal: test whether `mlp_moe16a2` wants fewer total parameters and larger batches than the first inherited dense-MLP-shaped baselines. All runs keep 16 experts, 2 active experts, and `expert_mlp_ratio = 2.0`. Most candidates also reduced the router auxiliary loss from `0.01` to `0.001`.

Metrics are post-hoc W&B tail means over the last 100 rows where both `loss/total` and `metrics/policy_top1` are present. The `1e16` runs were stopped before full target compute, so they are useful directional signals but are not replacement baselines.

| Budget | Status | Shape | Batch | LR | Params | Compute Seen | Loss | Policy Top-1 | Runtime | W&B |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `1e14` | finished | `d24x1` | 512 | 6.1e-4 | 274,286 | 1.00e14 | 3.9657 | 0.2232 | 5.9m | https://wandb.ai/maxence-frenette/chess-engine-4/runs/e8tlqogk |
| `1e14` | finished | `d32x1` | 512 | 6.1e-4 | 389,670 | 1.00e14 | 3.9371 | 0.2243 | 4.5m | https://wandb.ai/maxence-frenette/chess-engine-4/runs/nt5yfokj |
| `1e14` | finished | `d40x1` | 768 | 5e-4 | 517,342 | 1.00e14 | 4.2033 | 0.2222 | 3.6m | https://wandb.ai/maxence-frenette/chess-engine-4/runs/4kuh6b6g |
| `1e14` | finished | `d48x1` | 1,024 | 4.3e-4 | 657,302 | 1.00e14 | 4.7117 | 0.2178 | 2.7m | https://wandb.ai/maxence-frenette/chess-engine-4/runs/p3f6s8mb |
| `1e15` | finished | `d48x2` | 1,024 | 5e-4 | 879,254 | 1.00e15 | 3.6534 | 0.2949 | 21.3m | https://wandb.ai/maxence-frenette/chess-engine-4/runs/kpq91xhm |
| `1e15` | finished | `d64x2` | 1,024 | 5e-4 | 1,368,326 | 1.00e15 | 3.5940 | 0.2999 | 16.6m | https://wandb.ai/maxence-frenette/chess-engine-4/runs/977dsu13 |
| `1e15` | finished | `d80x2` | 1,536 | 4.08e-4 | 1,955,702 | 1.00e15 | 3.6144 | 0.2996 | 15.4m | https://wandb.ai/maxence-frenette/chess-engine-4/runs/3vtgwrno |
| `1e15` | finished | `d64x3` | 1,536 | 4.08e-4 | 1,762,566 | 1.00e15 | 3.6187 | 0.3010 | 19.2m | https://wandb.ai/maxence-frenette/chess-engine-4/runs/hsrjvwxt |
| `1e15` | finished | `d40x2` | 1,024 | 5e-4 | 671,582 | 1.00e15 | 3.6420 | 0.2786 | 24.9m | https://wandb.ai/maxence-frenette/chess-engine-4/runs/ifhxaty9 |
| `1e15` | finished | `d48x1` | 1,024 | 5e-4 | 657,302 | 1.00e15 | 3.6506 | 0.2887 | 19.5m | https://wandb.ai/maxence-frenette/chess-engine-4/runs/ssn3dly0 |
| `1e16` | partial | `d96x4` | 4,096 | 1.5e-4 | 4,413,926 | 8.57e15 | 3.4409 | 0.3388 | 107.2m | https://wandb.ai/maxence-frenette/chess-engine-4/runs/z40swcyl |
| `1e16` | partial | `d128x4` | 4,096 | 1.5e-4 | 7,457,478 | 3.42e15 | 3.6685 | 0.3057 | 46.8m | https://wandb.ai/maxence-frenette/chess-engine-4/runs/lwfxsl5d |

## Takeaways

- `1e14`: none of the new runs beat the existing baseline tail loss (`3.9048`). The best new tail loss was `d32x1` at `3.9371`; larger batch/lower LR hurt here.
- `1e15`: `d64x2` is the best new run with tail loss `3.5940`, improving the current baseline (`d80x3`, `3.6246`) while using about half the total parameters.
- `1e16`: `d96x4` was promising but partial. At 85.7% of target compute it reached tail loss `3.4409`, still behind the completed baseline (`d160x6`, `3.3726`). It is a useful lower-parameter candidate for a full rerun, especially with more conservative runtime expectations.
- The `d20x1` follow-up failed before training because bf16 `grouped_mm` requires the row stride to be 16-byte aligned; practically, keep `d_model` as a multiple of 8.

## Baseline Updates

Only `1e15` was updated from this sweep: `d64x2`, batch `1024`, LR `5e-4`, router aux `0.001`. The `1e14` and `1e16` baselines stay unchanged until a completed run beats them by the tail-loss metric.
