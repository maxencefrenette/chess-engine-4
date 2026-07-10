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

- `[model]`: architecture shape and model kind.
- `[data]`: batch size and loader settings.
- `[optimizer]`: learning rate, weight decay, and optimizer implementation
  details.

Use `[infra]` for cost optimization, not model-quality tuning. The goal of the
infra section is to reduce the dollar cost and wall-clock time of a full run
given the rest of the config. Examples include GPU type and future hardware
allocation settings.

## Transformer Engine

Training is CUDA-only and uses NVIDIA Transformer Engine for dense linear,
RMSNorm, fused dense SwiGLU MLP, and grouped MoE expert operations. The packed
input and dense model are captured with TE's high-level CUDA graph API. MoE
keeps only the packed-plane expansion under `torch.compile`: TE's BF16 grouped
expert path reads dynamic expert split sizes on the host, so the complete MoE
model is not CUDA-graph safe. The fully fused graph-safe grouped expert kernel
is currently available for TE's MXFP8 and NVFP4 recipes, not BF16.

Modal training reserves eight physical CPU cores and the baseline configs use
eight Rust dataloader threads. This keeps batch decoding ahead of the captured
dense model, where four loader threads became the end-to-end bottleneck.
