# Dataloader optimization pass

This pass tested the PyTorch data-loading tutorial recommendations against the MLP `1e19` baseline for 500 steps on Modal. Batch size tuning and `in_order=False` were intentionally skipped. All runs used an L4 GPU, W&B disabled, and a temporary `cpu=4` Modal patch so worker-count tests had enough CPU allocation.

Reference: https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html

## Result

No code change was kept. The current default, `num_workers=0` with PyTorch `DataLoader(batch_size=None, pin_memory=True)`, was the fastest non-noisy option.

| Candidate | Final step ms | Avg step ms, 400-500 | Outcome |
| --- | ---: | ---: | --- |
| Baseline: `num_workers=0`, pinned memory | 96.5 | 95.6 | Keep current behavior |
| `num_workers=1` | 389.4 | 380.8 | Reject |
| `num_workers=2` | 222.2 | 226.7 | Reject |
| `num_workers=4` | 209.3 | 219.7 | Reject |
| `pin_memory=False` | 95.2 | 95.1 | Reject as noise |
| CUDA stream prefetch | 152.8 | 148.6 | Reject |
| `num_workers=4`, `file_system` sharing | 287.2 | 275.4 | Reject |
| Direct dataset iteration, no DataLoader | 112.9 | 107.5 | Reject |

## Interpretation

The loader is already doing the dataset-level batching that matters for this workload: `LeelaTarDataset` yields fully formed tensor batches and the PyTorch DataLoader is configured with `batch_size=None`, so there is no per-sample Python collation path to optimize.

Multiprocessing workers are bad here because each worker returns large already-batched tensors. The cost of moving those tensors through worker IPC and shared memory is larger than the benefit of background tar parsing. `file_system` sharing did not fix this, which points away from file-descriptor pressure and toward large-tensor transfer overhead.

Pinned memory and non-blocking transfer were already enabled in the baseline. Disabling pinning was about 0.5 ms/step faster in this single run, but that is too small to justify changing the training path. The explicit CUDA stream prefetcher was much slower, so H2D transfer overlap is not the missing bottleneck in the current setup.

`persistent_workers=True` remains harmless but inactive for the default path because `num_workers=0`. With workers enabled, it only avoids worker restart overhead across epochs; these training runs consume one long iterable stream, so there is no meaningful epoch-boundary benefit to measure.

`__getitems__` does not directly apply to `IterableDataset`. PyTorch's batched `__getitems__` optimization is for map-style datasets where the DataLoader requests a list of indices. Our equivalent is already implemented by yielding complete batches from `__iter__`.
