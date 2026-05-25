# Training Speed Optimization

Target: per-step training speed for `configs/mlp_moe16a2/1e19.toml` on Modal L4. Short runs used 200 steps to keep benchmark cost low.

Baseline average interval step time from two true baseline runs: `0.1248s`.

Best retained interval step time after restoring diagnostics: `0.1152s`.

Net per-step speedup: `7.7%`.

## Kept Changes

| Change | Best evidence | Interval step time | Individual outcome | Notes |
| --- | --- | ---: | ---: | --- |
| Add `--max-steps` | CLI behavior | n/a | n/a | No speed effect; keeps short benchmarks exact and reusable. |
| Compile CUDA training model | `opt-1e19-pytorch-pin-workers0-200` | `0.1152s` | `+7.7%` vs baseline avg | Kept. Compile is not restricted by model kind; for MoE, the grouped expert path is excluded because full grouped_mm compile was slower. |

The best retained current-shape run is [opt-1e19-pytorch-pin-workers0-200](https://wandb.ai/maxence-frenette/chess-engine-4/runs/z55cuaj1), `0.1152s/step`. The older `0.1021s/step` compile run removed diagnostics and is no longer the retained code shape.

## Profiling

Temporary synchronized phase timing on Modal L4 for a 50-step run showed:

| Phase | Avg sec/step |
| --- | ---: |
| Data loading / batch prep | `0.0734` |
| Host-to-device move | `0.0117` |
| Forward + loss | `0.0270` |
| Backward | `0.0195` |
| Optimizer | `0.0024` |
| Logging | `0.0002` |

The main bottleneck is CPU-side data preparation. The GPU compute path is not the dominant wall-clock cost yet.

Local loader profiling agreed: `planes_from_frames` and gzip decompression dominate batch construction. Loader micro-optimizations were reverted because the wins were small/noisy relative to the code complexity. The bigger remaining win is probably changing the data representation or moving plane expansion to GPU.

## Prefetch Tests

Only PyTorch-native prefetch/pinning APIs are retained in the codebase. The custom-thread experiments were reverted even when fast, because they bypassed `DataLoader`'s prefetch implementation.

| Attempt | Interval step time | Outcome |
| --- | ---: | --- |
| PyTorch `DataLoader(num_workers=1, prefetch_factor=2)` | `0.3104s` | Reverted; slower. |
| PyTorch `DataLoader(num_workers=2, prefetch_factor=2)` | `0.3642s` | Reverted; slower. |
| PyTorch `pin_memory=True`, `num_workers=0` | `0.1152s` | Reverted; slower than the retained no-pin path. |
| PyTorch `pin_memory=True`, `num_workers=1` | `0.5349s` | Reverted; much slower. |
| Custom background thread prepares CPU batch; main thread copies to GPU | `0.0940s`, `0.0767s` | Reverted; fast, but not kept because this reimplemented prefetching outside PyTorch's `DataLoader` APIs. |
| Background thread pins memory and enqueues H2D copy | `0.0890s`, `0.2670s` | Reverted; one fast run, one very slow run. |
| `cpu=2` plus pinned background H2D copy | `0.1427s` | Reverted; slower. |
| Background thread enqueues H2D copy without pinning | `0.0825s`, `0.1634s` | Reverted; unstable. |
| `cpu=2` plus no-pin background H2D copy | `0.1563s` | Reverted; slower. |

This suggests the useful overlap is CPU-side LC0 batch construction, but PyTorch's process-based `DataLoader` prefetch is a poor match for the current dataset because it ships already-built full batches through multiprocessing queues. Cross-thread CUDA transfer was too variable on Modal for this setup, and requesting `cpu=2` did not improve it.

## Worker Tests

The background-worker hypothesis did not hold on Modal for this workload.

| Attempt | Interval step time | Outcome |
| --- | ---: | --- |
| `num_workers=1`, default CPU | `0.3092s` | Reverted. |
| `num_workers=2`, default CPU | `0.2063s` | Reverted. |
| `num_workers=4`, default CPU | `0.2020s` | Reverted. |
| `cpu=2`, `num_workers=0` | `0.2048s` | Reverted. |
| `cpu=2`, `num_workers=1` | `0.4237s` | Reverted. |
| `cpu=2`, `num_workers=2` | `0.2548s` | Reverted. |

The likely issue is that the dataset already yields complete batches, so worker processes must ship very large tensors through multiprocessing queues. That IPC cost overwhelms any overlap.

## Compile Tests

| Attempt | Runtime | Outcome |
| --- | ---: | --- |
| Full `torch.compile(model)` | failed | Dynamo could not fake-trace the `grouped_mm` path under autocast. |
| `torch.compile(lczero_loss)` | `34.291s` | Reverted; overhead exceeded kernel savings. |
| `torch.compile` dense submodules only | `36.174s` | Reverted. |
| Compile model while disabling the expert path | `0.1021s/step` | Kept for CUDA MoE training. Total 200-step runtime was `33.086s`, so use longer runs when judging this path. |
| Full grouped_mm compile, `max-autotune-no-cudagraphs`, explicit BF16 operands | `0.1255s/step` | Reverted; it ran, but was slower than the graph-break expert path. |
| Full grouped_mm compile, default mode, explicit BF16 operands | `0.1740s/step` | Reverted; slower. |
| Compile while disabling expert path, explicit BF16 operands | `0.1779s/step` | Reverted; the explicit cast made the fast graph-break path much slower. |
| Full BF16 parameters only, full grouped_mm compile | failed | Reverted; compiled grouped_mm still saw FP32 activations under autocast. |
| Full BF16 parameters and inputs, no autocast | failed | Reverted; compiled forward worked, then backward failed on a float32 target/loss dtype mismatch. |
| Full BF16 parameters, inputs, targets, no autocast | `0.1208s/step` | Reverted; it worked end to end, but was slower than the graph-break compile path. |

Compile is now kept because the optimization target is steady-state per-step time. It should be amortized over normal training runs.

Notes from the follow-up grouped_mm compile probe:

- PyTorch documents `torch.nn.functional.grouped_mm` as a CUDA SM80+ BF16 grouped GEMM API for MoE-style jagged expert batches.
- Hugging Face's MoE backend docs say compiled `grouped_mm` is compatible in BF16 and should use `mode=None` or `mode="max-autotune-no-cudagraphs"` because it is not compatible with CUDA graphs.
- That fixed the original dtype failure once operands were explicitly BF16, but for this model shape it did not beat keeping the grouped expert kernel outside Dynamo and compiling the rest of the model with `reduce-overhead`.
- Full BF16 training also works if parameters, inputs, and targets are all BF16 and autocast is disabled. That gets grouped_mm inside the compiled graph without the fake-tensor dtype crash, but it measured slower than the current retained path.

## Other Failed Or Neutral Attempts

| Attempt | Runtime | Outcome |
| --- | ---: | --- |
| CPU bf16 plane conversion before transfer | `66.208s` | Reverted; CPU conversion was too expensive. |
| `torch.no_grad()` around metric construction | `29.829s` | Reverted; slower. |
| Remove grad norm only | `27.506s` | Reverted as a standalone change; no speedup. |
| Lean W&B logging / removed diagnostics | `0.1247s/step` | Reverted; neutral alone, and the lost observability was not worth it. |
| Data loader `deque`, streaming tar, and chunk-fill micro-optimizations | `0.1150s-0.1209s/step` | Reverted; small/noisy wins and added complexity. |
| Disable pinned memory | `0.1172s/step` | Reverted; after restoring diagnostics this was not meaningfully better than PyTorch's standard pinned-memory path. |
| No pinned memory with full logging | `61.214s` | Reverted; bad interaction. |
| Remove explicit CUDA sync for logging intervals | `29.529s` | Reverted; worse and made timing less reliable. |
| Router bookkeeping with `bincount` instead of `one_hot` | `39.070s` | Reverted; slower on CUDA despite lower apparent allocation. |

## Next Targets

The next meaningful optimization work should focus on the data representation:

- Avoid constructing full float32 planes on CPU.
- Consider transferring packed bitboards and expanding on GPU.
- Reduce gzip/tar overhead by pre-extracting or repacking Modal volume data into larger contiguous chunks.
