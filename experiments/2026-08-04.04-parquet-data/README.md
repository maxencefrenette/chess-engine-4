# Parquet training data migration

This experiment replaced the LCZero tar training corpus with a Rust/Polars
Parquet pipeline. The format retains packed input planes, the six non-constant
plane scalars, compact FP16 policy targets, and root Q/D/M. Training now reads
Parquet exclusively.

## Data

All 480 Modal training files were converted into `/parquet` on the existing
training-data Volume. The corpus shrank by 57.8%, from 755.5 GiB of tar files to
319.2 GiB of Parquet files. Conversion ran in 16 resumable workers and typically
processed 35,000-50,000 positions/s per worker. One truncated source archive was
detected, downloaded again, and converted successfully.

## Correctness

The local real-data comparison and the initial Modal comparison matched exactly.
After full conversion, one independently decoded batch from every one of the 480
tar/Parquet pairs matched exactly for packed planes, plane scalars, compact
policy indices, FP16 policy bits, and root Q/D/M. The Parquet files intentionally
do not retain unused value targets.

## Loader throughput

A one-worker local benchmark including Parquet decoding, fixed-batch assembly,
NumPy ownership transfer, and `torch.from_numpy` reached 1.05M positions/s,
versus 0.20M positions/s for tar, a 5.3x loader-level improvement.

Final matched B200 profiles used the canonical recipe and the following commands:

```sh
uv run profile-training --d-model 512 --warmup-steps 50 --profile-steps 1000 --json
uv run profile-training --d-model 1024 --warmup-steps 50 --profile-steps 1000 --json
uv run profile-training --d-model 2048 --warmup-steps 30 --profile-steps 500 --json
```

| Width | Format | Step time | Fetch wall | Exposed GPU idle |
| --- | --- | ---: | ---: | ---: |
| d512 | tar | 22.30 ms | 16.92 ms | 14.29 ms |
| d512 | Parquet | 7.14 ms | 0.11 ms | 0.12 ms |
| d1024 | tar | 42.28 ms | 33.32 ms | 19.25 ms |
| d1024 | Parquet | 22.88 ms | 0.09 ms | 0.12 ms |
| d2048 | tar | 111.80 ms | 0.92 ms | 0.18 ms |
| d2048 | Parquet | 112.38 ms | 0.39 ms | 0.17 ms |

Parquet reduced end-to-end step time by 68.0% at d512 and 45.9% at d1024 by
removing exposed loader stalls. At d2048, which is GPU-bound, step time differed
by only 0.5%. A post-cutover d512 smoke profile completed at 7.52 ms/step.

After the exact corpus verification, sustained profiles, full test suite, and
post-cutover smoke test passed, all 480 source tar files were deleted from the
Modal Volume. The retained Parquet corpus occupies 319.2 GiB.
