# Dataloader optimization pass

This pass tested the PyTorch data-loading tutorial recommendations against the MLP `1e19` baseline for 500 steps on Modal. Batch size tuning and `in_order=False` were intentionally skipped. All runs used an L4 GPU, W&B disabled, and a temporary `cpu=4` Modal patch so worker-count tests had enough CPU allocation.

Reference: https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html

## Updated Result

The worker and prefetch knobs did not help, but packed input planes did. Training now asks `LeelaTarDataset` for packed bitboard bytes plus the eight scalar planes for every model family and device. Dense LC0 planes remain the core model/export interface, and a training-only wrapper unpacks the packed representation before calling the core model.

| Candidate | Final step ms | Avg step ms, 400-500 | Outcome |
| --- | ---: | ---: | --- |
| Baseline: `num_workers=0`, pinned memory | 96.5 | 95.6 | Previous baseline |
| `num_workers=1` | 389.4 | 380.8 | Reject |
| `num_workers=2` | 222.2 | 226.7 | Reject |
| `num_workers=4` | 209.3 | 219.7 | Reject |
| `pin_memory=False` | 95.2 | 95.1 | Reject as noise |
| CUDA stream prefetch | 152.8 | 148.6 | Reject |
| `num_workers=4`, `file_system` sharing | 287.2 | 275.4 | Reject |
| Direct dataset iteration, no DataLoader | 112.9 | 107.5 | Reject |
| CPU planes as float16 | 261.7 | 267.3 | Reject |
| Dense CPU float32 planes copied as bf16 | 99.5 | 100.1 | Reject |
| Split `uint8` binary planes + scalar planes | 67.2 | 68.8 | Previous keep |
| Packed bytes + GPU bitshift unpack | 92.2 | 92.6 | Reject |
| Packed bytes + GPU lookup-table unpack | 71.8 | 71.7 | Reject |
| Sparse policy, CPU cumsum ranks | 104.5 | 103.8 | Reject |
| Sparse policy, compact rank construction | 90.1 | 87.7 | Reject |
| Packed bytes + compiled model unpack | 62.4 | 60.5 | Previous keep |
| Packed bytes + compiled training wrapper | 60.7 | 57.7 | Keep |
| Compact policy K=218, NumPy nonzero | 84.0 | 80.9 | Reject |
| Compact policy K=218, flat NumPy | 112.2 | 85.5 | Reject |
| Compact policy K=218, Torch CPU | 101.6 | 103.6 | Reject |
| Dense policy float16 | 108.8 | 105.5 | Reject |
| Dense policy float16 copied as bf16 | 90.9 | 85.9 | Reject |
| Dense policy CPU bf16 | 75.8 | 81.8 | Reject |
| Rust native loader | 39.2 | 37.4 | Keep |
| Rust compact policy fp16 | 31.5 | 31.9 | Keep |

The compact-policy native loader improves the final measured interval by about 67% versus the original baseline and about 15% versus the dense-policy Rust native loader.

## Interpretation

The loader is already doing the dataset-level batching that matters for this workload: `LeelaTarDataset` yields fully formed tensor batches and the PyTorch DataLoader is configured with `batch_size=None`, so there is no per-sample Python collation path to optimize.

Multiprocessing workers are bad here because each worker returns large already-batched tensors. The cost of moving those tensors through worker IPC and shared memory is larger than the benefit of background tar parsing. `file_system` sharing did not fix this, which points away from file-descriptor pressure and toward large-tensor transfer overhead.

Pinned memory and non-blocking transfer were already enabled in the baseline. Disabling pinning was about 0.5 ms/step faster in this single run, but that is too small to justify changing the training path. The explicit CUDA stream prefetcher was much slower, so H2D transfer overlap is not the missing bottleneck in the current setup.

`persistent_workers=True` remains harmless but inactive for the default path because `num_workers=0`. With workers enabled, it only avoids worker restart overhead across epochs; these training runs consume one long iterable stream, so there is no meaningful epoch-boundary benefit to measure.

`__getitems__` does not directly apply to `IterableDataset`. PyTorch's batched `__getitems__` optimization is for map-style datasets where the DataLoader requests a list of indices. Our equivalent is already implemented by yielding complete batches from `__iter__`.

The first compact-plane attempt used CPU `float16` planes. That was much slower, likely because CPU-side float16 materialization and fp16-to-bf16 conversion outweighed transfer savings. Keeping dense CPU float32 planes but requesting bf16 on the CUDA copy also did not help. The first useful version was the split representation: keep binary planes byte-sized until they are on GPU, and only transfer scalar plane values separately.

Packed-byte transfer reduced host-to-device bytes further. The first two packed-byte versions unpacked before the compiled model call, so the extra GPU unpack work ran eagerly and was too expensive. Moving the unpack into a compiled training wrapper changed the result for the MLP benchmark: it now beats the split `uint8` representation while keeping core models as dense LC0-plane modules. This suggests the bandwidth reduction is real, but only pays off when the unpack is part of the compiled input path.

Sparse policy transfer was also slower. The first version built padded policy indices with a full CPU cumsum over the dense policy matrix and was much worse. A tighter rank-construction version reduced the overhead, but still lost to the dense policy path. For this workload, the saved transfer bytes do not pay for the extra CPU sparse construction plus the gathered sparse cross-entropy path. PyTorch sparse tensors are unlikely to help here because the loss is row-wise indexing and reduction, not sparse matrix algebra; they would add sparse tensor construction and layout overhead without matching the kernel shape we need.

Fixed-width compact policy with `K=218` was faster on the GPU loss in a synthetic benchmark, but slower end to end. The missing piece is host-side construction: online dense-to-compact conversion from LC0 records costs more than the transfer savings. This is still a good candidate for a future Rust/preprocessed dataloader, where policy compaction can happen while decoding records without Python or NumPy `nonzero` overhead.

Dense half-precision policy transfer also underperformed. CPU-side `float16` materialization was cheap, but the full training path slowed down substantially. Converting the fp16 tensor to bf16 on-device helped but still lost. Building CPU bf16 policy tensors also lost.

## CPU/GPU Breakdown

A follow-up Modal profile on the best Python path used `configs/mlp/1e19.toml`, L4, batch size 4096, packed-plane input, dense float32 policy, `num_workers=0`, and `pin_memory=True`. It measured 200 steps after 50 warmup steps.

| Bucket | Mean |
| --- | ---: |
| Total wall time | 70.55 ms/step |
| CPU/dataloader fetch wall | 65.81 ms/step |
| Exposed GPU idle gap | 62.71 ms/step |
| H2D copy on GPU stream | 2.67 ms/step |
| Train GPU kernels | 5.15 ms/step |
| Copy + train GPU work | 7.82 ms/step |

This profile says the current path is strongly CPU/dataloader-bound. The GPU does useful copy plus training work for roughly 11% of the step wall time, while the exposed idle gap is roughly 89%. Train-only throughput was about 17.3 TFLOP/s, or 14.3% MFU on L4 bf16; end-to-end MFU is much lower because the GPU mostly waits for the next batch.

## Native Loader Follow-Up

The first Rust loader keeps the same batch contract as the Python loader: packed planes, scalar planes, dense float32 policy, and value targets. It is exposed through `pyo3`/`maturin`, returns NumPy arrays to Python, and the training loop wraps those arrays with `torch.from_numpy`.

A 500-step Modal benchmark on the same MLP `1e19` setup reached 39.2 ms/step on the final interval and 37.4 ms/step averaged over steps 400-500. That is about 35% faster than the previous best packed-plane Python loader at 57.7 ms/step. The native loader is now the only training dataloader.

## Native CPU/GPU Breakdown

A reusable Modal profiler was added as `uv run profile-training`. Running

```sh
uv run profile-training --config configs/mlp/1e19.toml --warmup-steps 50 --profile-steps 200
```

on the native loader produced this breakdown:

| Bucket | Mean |
| --- | ---: |
| Total wall time | 45.76 ms/step |
| CPU/dataloader fetch wall | 37.86 ms/step |
| Python enqueue wall | 4.05 ms/step |
| Exposed GPU idle gap | 37.97 ms/step |
| H2D copy on GPU stream | 2.65 ms/step |
| Train GPU kernels | 5.13 ms/step |
| Copy + train GPU work | 7.79 ms/step |

The native loader substantially reduced host-side overhead, but training is still mostly input-bound. The GPU performs copy plus training work for about 17% of wall time, with about 83% exposed idle gap. Train-only MFU was 14.4% on L4 bf16, while end-to-end MFU was 1.6%.

## Compact Policy Follow-Up

The Rust loader now emits policy targets as `policy_indices: int16 [B, 218]` and `policy_probs: float16 [B, 218]`, padded with `-1` indices and zero probability. The policy loss gathers logits at legal move indices and does the soft-label cross entropy over the compact legal-move set. The stored fp16 probabilities are cast back to fp32 for the loss reduction.

A 500-step Modal benchmark reached 31.5 ms/step on the final interval and 31.9 ms/step averaged over steps 400-500. The profiler breakdown was:

| Bucket | Mean |
| --- | ---: |
| Total wall time | 34.23 ms/step |
| CPU/dataloader fetch wall | 29.05 ms/step |
| Python enqueue wall | 4.04 ms/step |
| Exposed GPU idle gap | 29.15 ms/step |
| H2D copy on GPU stream | 0.65 ms/step |
| Train GPU kernels | 4.43 ms/step |
| Copy + train GPU work | 5.08 ms/step |

This is a clear win. H2D copy fell from 2.65 ms/step to 0.65 ms/step, while total profiled wall time fell from 45.76 ms/step to 34.23 ms/step. Training is still input-bound, but dense policy transfer is no longer the dominant payload.

## Threaded Native Prefetch Follow-Up

The PyTorch `DataLoader` wrapper was removed. The Rust loader now owns background prefetching directly: worker threads share one tar-file queue, each worker has a bounded ready-batch queue, and Python polls the worker queues for full batches. The current config uses `dataloader_threads = 4` and `dataloader_prefetch_per_thread = 2`.

The latest Modal profile used:

```sh
uv run profile-training --config configs/mlp/1e19.toml --dataloader-threads 4 --warmup-steps 50 --profile-steps 200
```

| Bucket | Mean |
| --- | ---: |
| Total wall time | 8.38 ms/step |
| CPU/dataloader fetch wall | 1.93 ms/step |
| Python enqueue wall | 5.38 ms/step |
| Exposed GPU idle gap | 2.04 ms/step |
| H2D copy on GPU stream | 1.65 ms/step |
| Train GPU kernels | 4.69 ms/step |
| Copy + train GPU work | 6.34 ms/step |

Compared with the single-thread compact-policy profiler at 34.23 ms/step, this cuts total profiled step time by about 75%. The remaining visible overhead is no longer batch fetch; Python enqueue plus host-to-device transfer is now a larger fraction of wall time. End-to-end MFU improved from 2.2% to 8.8% on the L4 profile.
