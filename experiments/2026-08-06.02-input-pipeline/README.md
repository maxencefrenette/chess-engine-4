# Input pipeline tuning

## Goal

Retune the host-to-device input path for dense widths that moved to Modal's
RTX PRO 6000 and establish an explicit per-width strategy for the `moe64a2`
family. The production training profiler used 50 warmup steps and between 100
and 1,000 measured steps depending on model size. W&B was disabled.

Each result includes the normal Rust/Polars Parquet loader, pinning or staging,
host-to-device transfer, CUDA graph, forward/backward pass, optimizer, and loss
logging path. Close dense results were repeated on fresh Modal workers.

Profiles used source commit `1888a0c0f51b6bf920ca3bb23060c82c14ad56d3`
with the candidate pipeline injected into the remote config payload. Data came
from `/data/training_data/parquet` on the canonical Modal training volume.

## Dense results

| Width | GPU | Previous | Selected | Previous ms/step | Selected ms/step | Reduction |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| d32 | RTX PRO 6000 | pageable | pageable | 2.22 | 2.22 | - |
| d64 | RTX PRO 6000 | pageable | pageable | 2.49 | 2.49 | - |
| d128 | RTX PRO 6000 | staging | pageable | 3.17 | 2.97 | **6.2%** |
| d256 | RTX PRO 6000 | staging | overlap | 4.09 | 3.98 | **2.7%** |

The table reports the mean of two independent profiles for each compared path.
At d32 through d128, copying pageable memory directly is cheaper than pinning
these small batches. At d256, the payload is large enough for pinned async H2D
on a separate stream to recover its setup overhead.

## MoE results

| Width | GPU | Previous | Selected | Previous ms/step | Selected ms/step | Reduction |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| d128 | RTX PRO 6000 | pinned | overlap | 10.86 | 10.33 | **4.8%** |
| d256 | RTX PRO 6000 | pinned | overlap | 33.44 | 32.52 | **2.8%** |
| d512 | B200 | pinned | overlap | 43.50 | 41.51 | **4.6%** |
| d1024 | B200 | pinned | overlap | 138.22 | 131.82 | **4.6%** |
| d2048 | B200 | pinned | overlap | 197.75 | 196.56 | **0.6%** |

The d128 through d1024 values are means of two independent profiles. The d2048
comparison used one profile at a reduced batch of 65,536 because the canonical
262,144 batch OOMed under every input pipeline; even a 131,072 batch only fit
with pageable transfers. The d2048 overlap selection follows the consistent
smaller-width trend and must be revalidated after its batch-size memory issue is
resolved.

## Conclusion

Dense models retain their existing width-specific strategy, with d128 moved to
`pageable` and d256 moved to `overlap`. All supported MoE widths now explicitly
select `overlap`. The input loader itself remained below 0.25 ms/step through
d256; the gains come from reducing exposed transfer and pinning time rather than
from changing Parquet loading.

Equivalent one-off profiling command:

```sh
uv run profile-training --config configs/moe64a2.py --d-model 256 \
  --warmup-steps 50 --profile-steps 500
```
