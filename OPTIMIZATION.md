# Optimization Notes

## Modified Compute

Experiments use an intentionally unusual modified-compute metric:

```text
modified_compute = flops_per_sample * batch_size * steps^2
```

The exponent is fixed at 2 and is not a hyperparameter. Higher step counts are
softly penalized. This reflects the fact that tiny steps are
not as efficient on real hardware as spending more of the budget on larger
batches or larger models. More steps mean more Python loop overhead, more
optimizer launches, more logging/checkpoint boundaries, and more small kernels
that fail to saturate the GPU.

This is not a claim that physical FLOPs stop mattering. Actual FLOPs are logged
separately as `perf/flops_seen`. Steps are configured directly; modified compute
is derived after choosing the model, batch size, and run length.

## Experiment Rules

When running a controlled experiment, keep `[loss]` fixed unless the training
target itself is under study. Keep the experiment interpretable and avoid changing
unrelated variables without a clear hypothesis.

Improve training efficiency by changing:

- `[model]`: architecture shape.
- `[run]`: batch size and steps.
- `[optimizer]`: learning rate, weight decay, and optimizer implementation
  details.

Use `[infra]` for cost optimization, not model-quality tuning. The goal of the
infra section is to reduce the dollar cost and wall-clock time of a full run
given the rest of the config. Examples include GPU type and future hardware
allocation settings.

## Transformer Engine

Training is CUDA-only and uses NVIDIA Transformer Engine for MLP layers. The
packed input and model are captured with TE's high-level CUDA graph API.

`[infra].cpu_cores` controls the physical CPU cores reserved by Modal, while
`[infra].dataloader_threads` controls the Rust batch-loading workers. Baseline
configs reserve eight cores and use eight workers; larger runs can reserve more
CPU headroom without changing the data-loading topology.
