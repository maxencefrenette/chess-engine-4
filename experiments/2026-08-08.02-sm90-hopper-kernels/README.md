# SM90 Hopper Kernels

## Goal

Add explicit H100/H200 training and standalone lc0 inference support without
changing canonical recipe GPU selections. Validate the first BF16 SM90 kernels
against Transformer Engine or a direct PyTorch reference, and retain custom
training only where it wins on the canonical end-to-end path.

This was focused correctness and benchmarking only. No training run, checkpoint,
or W&B run was created, so `EG_flops` is not applicable.

## Revisions and hardware

- Project base commit: `a78e1c6`.
- SM90 implementation commit: `4b8ab1600c52b417041f9c86ac833d8080a568db`.
- ThunderKittens: `1c3920d993404dd49a6d4c7267ea11d583bd5c68`.
- Transformer Engine reference: `8260f49660cbadb78bc52c90449428c51625469d`.
- H100 requests used Modal `H100!`; returned device: `NVIDIA H100 80GB HBM3`.
- H200 returned device: `NVIDIA H200`.
- Training cost includes the GPU and eight reserved CPU cores.

## Commands

```bash
uv run benchmark-training-modal --config configs/dense.py --d-model 128 --gpu H100 --level layer --warmup 2 --iterations 5 --json
uv run benchmark-training-modal --config configs/dense.py --d-model 128 --gpu H100 --level production --warmup 3 --iterations 10 --json
uv run benchmark-moe-kernels-modal --custom-gpu H100 --d-model 128 --warmup 2 --iterations 5 --custom-only --json
uv run benchmark-training-modal --config configs/moe64a2.py --d-model 128 --gpu H100 --level production --warmup 3 --iterations 10 --json
uv run benchmark-training-modal --config configs/dense.py --d-model 128 --gpu H200 --level layer --warmup 2 --iterations 5 --json
uv run benchmark-moe-kernels-modal --custom-gpu H200 --d-model 128 --warmup 2 --iterations 5 --custom-only --json
uv run prepare-lc0-modal --gpu H100
uv run benchmark-lc0-modal /tmp/ce4-sm90-smoke/sm90-dense-d128.safetensors --gpu H100 --batch-size 256 --batches 5
uv run benchmark-lc0-modal /tmp/ce4-sm90-smoke/sm90-moe64a2-d128.safetensors --gpu H100 --batch-size 256 --batches 5
uv run benchmark-lc0-modal /tmp/ce4-sm90-smoke/sm90-dense-d128.safetensors --gpu H200 --batch-size 256 --batches 5
uv run benchmark-lc0-modal /tmp/ce4-sm90-smoke/sm90-moe64a2-d128.safetensors --gpu H200 --batch-size 256 --batches 5
```

The first H100 production attempt, Modal run
`ap-0kl5LSD1rlnqVCcYn7LNCy`, stopped before timing because the Rust loader's
five-second cold-prefetch deadline expired. The benchmark harness now retries
that specific bounded startup timeout; the successful retry used the same
canonical loader and input pipeline.

## Numerical correctness

Dense results use Transformer Engine BF16 as the reference. MoE results use the
direct BF16 expert implementation as the reference.

```json
{
  "h100_dense": {
    "modal_run": "ap-E57hE5HG5Xwu2r2Ct2xfk4",
    "output": {"cosine_similarity": 1.0, "mean_absolute_error": 0.0, "max_absolute_error": 0.0},
    "gradient_cosine_min": 0.9999998807907104,
    "all_gradients_finite": true,
    "custom_forward_ms": 0.09328000247478485,
    "te_forward_ms": 0.03484800085425377,
    "custom_backward_ms": 0.11609599739313126,
    "te_backward_ms": 0.1738239973783493
  },
  "h100_moe": {
    "modal_run": "ap-MBEGyZkv1BmYyKhA8BwnxF",
    "output": {"cosine_similarity": 0.9999881982803345, "mean_absolute_error": 0.0009725235868245363, "max_absolute_error": 0.017578125},
    "gradient_cosine_min": 0.9999854564666748,
    "custom_forward_ms": 0.22579200565814972,
    "custom_backward_ms": 0.7117440104484558
  },
  "h200_dense": {
    "modal_run": "ap-bFpNosshrQZwIkQ6T1klgz",
    "output": {"cosine_similarity": 1.0, "mean_absolute_error": 0.0, "max_absolute_error": 0.0},
    "gradient_cosine_min": 0.9999998807907104,
    "all_gradients_finite": true,
    "custom_forward_ms": 0.09027200192213058,
    "te_forward_ms": 0.030912000685930252,
    "custom_backward_ms": 1.3808319568634033,
    "te_backward_ms": 0.14560000598430634
  },
  "h200_moe": {
    "modal_run": "ap-DzjjlBHjLJ8thHJt2klRAW",
    "output": {"cosine_similarity": 0.9999881982803345, "mean_absolute_error": 0.0009725235868245363, "max_absolute_error": 0.017578125},
    "gradient_cosine_min": 0.9999854564666748,
    "custom_forward_ms": 0.22681599855422974,
    "custom_backward_ms": 0.7103999853134155
  }
}
```

All outputs and gradients passed the repository's preserved acceptance
thresholds: dense output cosine at least `0.999`, mean absolute error at most
`1e-3`, and gradient cosine at least `0.99`; MoE output cosine at least `0.999`
and gradient cosine at least `0.99`.

## End-to-end H100 training benchmark

Both implementations used the same model state, canonical batch size, BF16
precision, pinned input pipeline, Parquet loader settings, eight reserved CPU
cores, and alternating paired measurement order.

```json
{
  "dense_d128": {
    "modal_run": "ap-XQfC4vuctduULm3DHMLtIc",
    "batch_size": 4096,
    "dollars_per_second": 0.0012018,
    "te": {"gpu_ms_median": 5.179424047470093, "wall_ms_median": 5.199654999999304, "dollars_per_step": 0.000006248945378999163},
    "custom": {"gpu_ms_median": 5.1976478099823, "wall_ms_median": 5.222725500000358, "dollars_per_step": 0.000006276671505900431},
    "custom_cost_efficiency_vs_te": 0.9955826703890425
  },
  "moe64a2_d128": {
    "modal_run": "ap-dDRIJ3AO9pX6xbLWB4bZRq",
    "batch_size": 16384,
    "dollars_per_second": 0.0012018,
    "te": {"gpu_ms_median": 24.255775451660156, "wall_ms_median": 24.312238500000305, "dollars_per_step": 0.000029218448229300366},
    "custom": {"gpu_ms_median": 18.100784301757812, "wall_ms_median": 18.161901500000077, "dollars_per_step": 0.000021826973222700092},
    "custom_cost_efficiency_vs_te": 1.3386394866198454
  }
}
```

## Standalone lc0 inference

`uv run prepare-lc0-modal --gpu H100` built and cached the SM90 binary at
`/artifacts/bin/lc0-sm90` in Modal run `ap-WxFI4O9ts7p0hQ016MrBnS`. The smoke
exports are valid depth-two d128 dense and alternating moe64a2 Safetensors
models. Their zero weights make this a runtime-contract and dispatch check, not
a playing-strength evaluation.

```json
{
  "h100_dense": {"modal_run": "ap-1eT8onXp5gU73EfUp48zHH", "device_name": "NVIDIA H100 80GB HBM3", "batch_size": 256, "batches": 5, "mean_nps": 409066, "mean_ms": 0.6258},
  "h100_moe64a2": {"modal_run": "ap-7RBf1jVCWtZDciT9f85sHv", "device_name": "NVIDIA H100 80GB HBM3", "batch_size": 256, "batches": 5, "mean_nps": 318041, "mean_ms": 0.8049},
  "h200_dense": {"modal_run": "ap-XyaqUfBLxL5D7j4Edkz2fg", "device_name": "NVIDIA H200", "batch_size": 256, "batches": 5, "mean_nps": 399151, "mean_ms": 0.6414},
  "h200_moe64a2": {"modal_run": "ap-9YUFtmSkzBw0QSL7bvKwLm", "device_name": "NVIDIA H200", "batch_size": 256, "batches": 5, "mean_nps": 298039, "mean_ms": 0.8589}
}
```

Five batches are sufficient for the required execution smoke but too few for a
canonical cost decision. H100 and H200 remain separate measurements.

## Verdict

- Keep Transformer Engine as the ordinary H100/H200 dense training backend.
  H100 custom dense was `0.9956x` as cost-efficient end to end despite a faster
  layer backward.
- Use custom BF16 for moe64a2 d128 only when H100 is explicitly selected. It
  passed numerical acceptance and was `1.3386x` as cost-efficient as TE on the
  canonical end-to-end path.
- Keep Transformer Engine as the default for H200 training pending a separate
  end-to-end cost benchmark. H200 custom kernels remain available only through
  an explicit supported custom configuration.
- Do not change the canonical dense or MoE recipe GPU maps. The separate
  H100/H200 cross-GPU cost task owns any future hardware promotion.
- Standalone lc0 SM90 inference is supported for both dense and moe64a2 exports
  on H100 and H200.
