# Optimization Notes

## `step_penalty_k`

`step_penalty_k` is intentionally unusual. The training budget is not plain
FLOPs when `k > 1`; it is a step-adjusted compute budget:

```text
compute_budget = flops_per_sample * batch_size * steps^k
```

With `k = 1`, this is ordinary physical training FLOPs. With `k > 1`, higher
step counts are softly penalized. This reflects the fact that tiny steps are
not as efficient on real hardware as spending more of the budget on larger
batches or larger models. More steps mean more Python loop overhead, more
optimizer launches, more logging/checkpoint boundaries, and more small kernels
that fail to saturate the GPU.

This is not a claim that physical FLOPs stop mattering. It is a pragmatic
budgeting convention for experiments where wall-clock cost and hardware
efficiency matter. Actual FLOPs are still logged separately as `perf/flops_seen`;
`compute_budget` is the optimization budget used to choose the number of steps.

## Experiment Rules

When running experiments to improve pretraining efficiency, keep `[run]` and
`[loss]` fixed. In particular, do not change the target signal, loss weights,
`compute_budget`, or `step_penalty_k` while comparing recipes within the same
budget.

Optimize final loss under the fixed `compute_budget` by changing:

- `[model]`: architecture shape.
- `[data]`: batch size and loader settings.
- `[optimizer]`: learning rate, weight decay, and optimizer implementation
  details.

Use `[infra]` for cost optimization, not model-quality tuning. The goal of the
infra section is to reduce the dollar cost and wall-clock time of a full run
given the rest of the config. Examples include GPU type and future hardware
allocation settings.

## Transformer Engine

Training is CUDA-only and uses NVIDIA Transformer Engine for linear layers,
RMSNorm, and fused dense SwiGLU blocks. The packed input and model are captured
with TE's high-level CUDA graph API.

`[infra].cpu_cores` controls the physical CPU cores reserved by Modal, while
`[infra].dataloader_threads` controls the Rust batch-loading workers. Baseline
configs reserve eight cores and use eight workers; larger runs can reserve more
CPU headroom without changing the data-loading topology.
