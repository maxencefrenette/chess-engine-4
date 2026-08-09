# Position subsampling for value diversity

## Status

Ready for authorization of three replacement runs. The exact startup snapshot, initial aborted-run
evidence, and successful eight-thread throughput probe are recorded below. No treatment run has
completed, no candidate checkpoint was exported, and no tournament, sampling-rule promotion, or
canonical data mutation has been performed by this task.

Task: `01kzg0rdwf`  
Branch: `codex/position-subsampling-01kzg0rdwf`  
Base commit: `316c327da2612f5b71e9776a9ae8d3fc773c216d`

## Design correction

The first capacity analysis incorrectly strengthened position retention into a within-game,
raw-archive materialization requirement. The user rejected that interpretation. The experiment
now reads the existing canonical Parquet shards directly and retains rows while streaming. It
does not inspect game boundaries, read raw tar archives, create derived datasets, or modify
canonical Parquet.

For each row, the sampler computes a stable 64-bit FNV-1a seed from the Parquet shard basename,
mixes the absolute zero-based row index with SplitMix64 finalization, and takes the result modulo
four. The treatments retain residues `< 4`, `< 2`, and `< 1`, respectively, for rates `1.0`,
`0.5`, and `0.25`. The quarter subset is therefore nested inside the half subset, which is nested
inside the full corpus. Decisions are independent of worker scheduling and global RNG state.

The first launch commands overrode the Parquet loader to one thread. With sorted shard paths, this
made the consumed row sequence and final partial-shard cutoff reproducible, not merely each row's
accept/reject decision. It also created a severe input bottleneck. The row decision itself remains
reproducible with multithreaded prefetch because it depends only on shard identity and row index,
as the user required.
Each shard still drops an incomplete final batch, matching the existing loader contract; the
footer audit reports both accepted rows and exact batch-usable rows so all treatments can use the
quarter arm's common step count without repetition.

The retention rate is recorded in the launch summary, W&B config, returned run metadata, and
checkpoint metadata. `train-modal --dry-run` prints the exact launch summary without starting
Modal, allowing all three summaries and costs to be reviewed before the authorized paid runs.

## Corpus stabilization gate

The experiment was originally redirected to the then-canonical snapshot of 480 shards and
3,949,735,220 rows. Before launch, a read-only footer audit observed 491 shards and 4,030,056,311
rows because it ran during an authorized concurrent corpus conversion. The corpus task then
reported 497 verified shards and 4,074,985,928 rows, with 17 unique new Parquet names and an exact
row delta matching the source audit. There is no overwrite or collision evidence; their temporary
source tars were deleted after exact verification.

The user intends the separate corpus task to continue filling verified Parquet up to the 900 GiB
operational ceiling, but clarified that training need not wait: the experiment freezes an explicit
startup filename manifest, and the planned iterator will not reach later atomic appends. This task
therefore uses the 497-shard snapshot recorded below and does not modify any corpus files.

## Startup snapshot and launch plan

The startup snapshot is retained in `parquet-files.txt`. It contains 497 sorted shard basenames and
has SHA-256 `ef7aa839e1ec2c01780123c42de086243995ec4cf9a7a0af3a0c01ffb7b3b595`. Later atomic
appends to `/parquet` cannot enter any treatment because every payload receives these exact 497
paths.

The footer and deterministic-hash audit produced:

| Rate | Accepted rows | Batch-usable rows |
| ---: | ---: | ---: |
| 1.0 | 4,074,985,928 | 4,058,644,480 |
| 0.5 | 2,037,509,235 | 2,021,195,776 |
| 0.25 | 1,018,779,501 | 1,003,094,016 |

The cost planner selected canonical `moe64a2 d512` in this sample range with 94% selection across
100 bootstrap fits. All treatments use the same 15,000 optimizer steps and therefore exactly the
same 983,040,000 accepted training samples. Retention only changes the number of underlying rows
that the loader scans to assemble each accepted batch: approximately 983,040,000 / 1,966,080,000 /
3,932,160,000 source rows for full / half / quarter retention. It never changes the batch size,
optimizer steps, or accepted sample allocation.

| Quantity | Matched value |
| --- | ---: |
| Model | moe64a2 d512x8, TE MXFP8 |
| Seed | 1 |
| Batch size | 65,536 |
| Steps | 15,000 |
| Samples | 983,040,000 |
| Training ratio | 0.0468065454273x |
| Learning rate | 0.00037 |
| FLOPs/sample | 181,555,318 |
| Training FLOPs | 1.7847613980672e17 |
| Loader threads | 1 |

Measured eight-thread full-retention steady-state time is 40.895007764 ms/step, including
0.398448734 ms/step of data fetch. For a deliberately conservative prelaunch bound, the fetch
component is scaled linearly by `8 / retention_rate` while the remaining step time is held fixed.
At the configured B200 plus eight-CPU rate of `$0.0018408/s`, the resulting bounds are:

| Rate | Bounded ms/step | Runtime | GPU+CPU cost |
| ---: | ---: | ---: | ---: |
| 1.0 | 43.684149 | 655.262 s | $1.206207 |
| 0.5 | 46.871739 | 703.076 s | $1.294222 |
| 0.25 | 53.246919 | 798.704 s | $1.470254 |

All three estimates are strictly below `$1.50`; the most conservative arm retains about three
cents of margin. The inspected dry-run launch summaries agree on every field except retention:

```sh
uv run train-modal --dry-run --config configs/moe64a2.py --d-model 512 \
  --training-ratio 0.04680654542731256 --steps 15000 --dataloader-threads 1 \
  --data-retention-rate RATE \
  --parquet-manifest experiments/2026-08-08.01-position-subsampling/parquet-files.txt \
  --wandb-name position-subsampling-rRATE
```

The three corresponding paid commands were launched concurrently for `RATE` values `1.0`, `0.5`,
and `0.25`. They were stopped as soon as live throughput proved that completing them would violate
the strict per-run cap.

## Aborted-run evidence and corrected cost model

The prelaunch estimate assumed that the measured eight-thread fetch time could be scaled to a
single thread and by inverse retention while the rest of the step remained unchanged. That model
was wrong: decoding and rejecting rows serialized the input pipeline and starved the GPU. The
three apps were stopped at the same wall-clock decision point, before any final checkpoint:

| Rate | Modal app | W&B | Last observed step | Representative samples/s | Projected cost to 15,000 steps |
| ---: | --- | --- | ---: | ---: | ---: |
| 1.0 | `ap-TAvG8K14mlB9iVyCxIEdJz` | [5e4itcmi](https://wandb.ai/maxence-frenette/uncategorized/runs/5e4itcmi) | 6,500 | ~0.83M | ~$2.18 |
| 0.5 | `ap-xpxRwoF129uYqToCVgFElh` | [fl24lk80](https://wandb.ai/maxence-frenette/uncategorized/runs/fl24lk80) | 1,680 | ~0.23M | ~$7.91 |
| 0.25 | `ap-DA8ZQ2oKGmUrOWnLv0pFgV` | [res2mpla](https://wandb.ai/maxence-frenette/uncategorized/runs/res2mpla) | 1,120 | ~0.145M | ~$12.49 |

The apps ran for roughly 8.5 minutes each. At `$0.0018408/s`, the provisional realized charge is
approximately `$0.94` per arm (`$2.82` total); exact Modal billing is not available in the training
result because termination prevented normal return. These are invalid partial runs and must not be
used for the retention comparison.

### Eight-thread quarter-rate throughput probe

The user authorized a small bounded probe before reducing experimental power. The inspected launch
summary specified eight CPU/loader threads, prefetch depth two per worker, retention 0.25, batch
65,536, and 500 steps / 32,768,000 accepted samples. Even at the prior one-thread throughput, its
training cost was bounded near `$0.42`; the actual Modal app lifetime was 76 seconds, or about
`$0.140` at the configured combined rate.

```sh
uv run train-modal --config configs/moe64a2.py --d-model 512 \
  --training-ratio 0.04680654542731256 --steps 500 \
  --dataloader-threads 8 --dataloader-prefetch-per-thread 2 \
  --data-retention-rate 0.25 \
  --parquet-manifest experiments/2026-08-08.01-position-subsampling/parquet-files.txt \
  --no-wandb --wandb-name position-subsampling-throughput-probe-r0.25
```

Modal app `ap-z7K9StZLBrgQXtNwiMy5JU` completed all 500 steps. After warmup, most logged intervals
sustained 1.59-1.60M accepted samples/s, matching the known eight-thread full-retention GPU-limited
rate. A few row-group/shard transitions dipped to 1.07-1.48M, but the complete 500-step training
window was about 21 seconds (~42 ms/step). Thus eight workers and the existing two-batch-per-worker
prefetch keep deterministic quarter-rate filtering off the GPU critical path as far as practical;
no loader code change or larger CPU allocation is needed.

Scaling the measured ~42 ms/step window to 15,000 steps gives about 630 seconds of training. Adding
the probe's conservative ~55 seconds of container startup and final-checkpoint overhead gives 685
seconds and `$1.261` per treatment. Because quarter retention is the worst scanning treatment and
it is GPU-fed, the full and half treatments should be approximately equal rather than materially
cheaper. The revised matched-sample plan therefore restores the original 15,000 steps / 983,040,000
accepted samples for all three arms, with an estimated cost of about `$1.26` each and roughly
`$0.24` margin below the cap.

The original authorization allowed exactly three bounded treatment runs. Because the first three
were started and aborted, no replacement treatment will be launched without explicit user approval.

## Evaluation protocol

If all three runs complete validly, export all checkpoints and run a connected searched lc0
tournament at approximately 800 nodes per move. Use mirrored openings and retain raw paired-game
evidence. Paired Elo/confidence-interval analysis should reuse the implementation from task
`01kzg0rdd6` if ready; this task will not independently rewrite that estimator. Report WDL, Elo and
CI, runtime, inference throughput, and the tournament cost estimate before launch, stopping for
approval if it exceeds `$1`.

## Current recommendation

Do not promote or reject a retention rate. Keep canonical configs and data unchanged and preserve
the sampler branch unmerged. The experiment can continue at full planned power once the user
authorizes three eight-thread replacement runs at matched 15,000 steps / 983,040,000 samples.
