# Dense half-batch throughput

## Goal

Measure the end-to-end production training throughput used to cost the adaptive
dense recipe when it selects the half-batch (`16d`) variant. The sweep preserves
the 1x sample count in its metadata by halving batch size and doubling steps.

## Run

Source commit: `f76410931a62649f79188d81659669dec441d2f8`, with the
uncommitted adaptive recipe and sweep support present in the worktree.

```sh
uv run throughput-sweep \
  --config configs/dense.py \
  --output experiments/throughput-dense-16d.toml \
  --widths 64 128 256 512 768 1024 1280 \
  --batch-divisor 2 \
  --warmup-steps 50 \
  --profile-steps 500
```

The initial d512 profile reported 17.11 ms/step and 3.8% end-to-end MFU, slower
than the full-batch profile. It was discarded as anomalous and refreshed with:

```sh
uv run throughput-sweep \
  --config configs/dense.py \
  --output experiments/throughput-dense-16d.toml \
  --widths 512 \
  --batch-divisor 2 \
  --warmup-steps 50 \
  --profile-steps 500 \
  --refresh
```

## Results

| Width | GPU | Batch | ms/step | Samples/s | End-to-end MFU |
| ---: | --- | ---: | ---: | ---: | ---: |
| d64 | RTX PRO 6000 | 1,024 | 2.705 | 378,542 | 0.46% |
| d128 | RTX PRO 6000 | 2,048 | 3.402 | 602,028 | 2.02% |
| d256 | RTX PRO 6000 | 4,096 | 3.388 | 1,208,807 | 12.63% |
| d512 | B200 | 8,192 | 7.142 | 1,147,021 | 9.21% |
| d768 | B200 | 12,288 | 7.537 | 1,630,389 | 13.93% |
| d1024 | B200 | 16,384 | 13.125 | 1,248,282 | 18.40% |
| d1280 | B200 | 24,576 | 23.276 | 1,055,836 | 23.89% |

The retained profiles represent about `$0.057` of steady-state GPU and CPU
time for 550 steps at each width. Startup and image-build time are excluded,
consistent with the budget planner's cost basis.

Integration preserved the newer exact-batch contract. D64 through d1024 are
exact `16d` profiles. The d1280 row was measured from the stale rounded recipe
at batch 24,576 (`19.2d`), so the planner does not use it for exact-`16d`
d1280 costing. The exact profiles cost about `$0.033`; the retained d1280
measurement accounts for about `$0.024` more.

## Verdict

Promote `experiments/throughput-dense-16d.toml` as the measured cost source for
matching half-batch dense configurations. Exact d1280 `16d` throughput remains
to be measured. The planner must match both width and selected batch size before
using a throughput row. This is a profiling experiment, so
W&B URL, validation loss, and `EG_flops` are not applicable.
