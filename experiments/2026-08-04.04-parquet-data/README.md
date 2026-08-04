# Parquet training data prototype

This experiment added an opt-in Rust/Polars Parquet converter and loader. The
format retains packed input planes, the six non-constant plane scalars, compact
FP16 policy targets, and root Q/D/M. Tar remains the default training format.

## Data

Three Modal training files were converted into `/parquet` on the existing
training-data Volume:

| Source | Tar size | Parquet size |
| --- | ---: | ---: |
| `20240401-0017` | 1.1 GiB | 460.7 MiB |
| `20240401-0117` | 1.7 GiB | 714.3 MiB |
| `20240401-0217` | 1.7 GiB | 725.5 MiB |

The fully measured first file contains 5,572,891 positions and shrank by 57.8%,
from 1,144,023,040 bytes to 483,120,959 bytes.

## Correctness

The local real-data comparison and a Modal comparison over four batches of
4,096 positions both matched exactly for packed planes, plane scalars, compact
policy indices, FP16 policy bits, and root Q/D/M. The Parquet file intentionally
does not retain unused value targets.

## Loader throughput

A one-worker local benchmark including Parquet decoding, fixed-batch assembly,
NumPy ownership transfer, and `torch.from_numpy` reached 1.05M positions/s,
versus 0.20M positions/s for tar, a 5.3x loader-level improvement.

Matched B200 profiles used the canonical recipe and the following commands:

```sh
uv run profile-training --d-model 512 --warmup-steps 50 --profile-steps 200 --json
uv run profile-training --d-model 512 --warmup-steps 50 --profile-steps 200 --parquet --json
uv run profile-training --d-model 1024 --warmup-steps 50 --profile-steps 100 --json
uv run profile-training --d-model 1024 --warmup-steps 50 --profile-steps 100 --parquet --json
uv run profile-training --d-model 2048 --warmup-steps 30 --profile-steps 50 --json
uv run profile-training --d-model 2048 --warmup-steps 30 --profile-steps 50 --parquet --json
```

| Width | Format | Step time | Fetch wall | Exposed GPU idle |
| --- | --- | ---: | ---: | ---: |
| d512 | tar | 12.81 ms | 7.18 ms | 5.39 ms |
| d512 | Parquet | 12.39 ms | 4.80 ms | 4.17 ms |
| d1024 | tar | 48.51 ms | 39.08 ms | 24.87 ms |
| d1024 | Parquet | 26.48 ms | 15.54 ms | 4.09 ms |
| d2048 | tar | 107.68 ms | 0.92 ms | 0.30 ms |
| d2048 | Parquet | 110.91 ms | 1.59 ms | 0.21 ms |

At d512, Parquet reduced step time by 3.3%. At d1024, it reduced step time by
45.4% and exposed idle by 83.5%. The d2048 result is GPU-bound under both
formats; its 3.0% slower Parquet wall time follows a 3.2% slower GPU-kernel
measurement, while loader idle remains negligible. There is no observed
loader-attributable regression.

Only three Parquet files were present, so at most three of the eight configured
workers had source files. The results therefore understate the steady-state
concurrency available after a full conversion.
