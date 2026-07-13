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

Keep the loss configuration fixed unless the training target itself is under
study. Model-quality experiments may change the model, run, and optimizer.
Infrastructure settings should only reduce cost or wall-clock time for an
otherwise identical run.

## Transformer Engine

Training is CUDA-only and uses NVIDIA Transformer Engine for MLP layers. The
packed input and model are captured with TE's high-level CUDA graph API.

`infra.cpu_cores` controls the physical CPU cores reserved by Modal, while
`infra.dataloader_threads` controls the Rust batch-loading workers. Baseline
configs reserve eight cores and use eight workers; larger runs can reserve more
CPU headroom without changing the data-loading topology.
