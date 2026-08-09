# Position subsampling for value diversity

## Outcome

Three matched `moe64a2 d512` treatments completed at retention rates 1.0, 0.5,
and 0.25. Each consumed exactly 983,040,000 accepted rows in 15,000 optimizer
steps. Half and quarter retention both improved training loss and searched-play
strength relative to full retention. Half retention has the best tournament
point estimate, but it is not distinguishable from quarter retention at this
sample size. Retain 0.5 as the leading candidate; do not promote it to a
canonical configuration until the user reviews this report.

Task: `01kzg0rdwf`  
Branch: `codex/position-subsampling-01kzg0rdwf`  
Base commit: `316c327da2612f5b71e9776a9ae8d3fc773c216d`

Relevant implementation commits are `3bbb430` (deterministic streaming
retention), `13dcaf0` (initial-run evidence), `7a51a48` (eight-thread probe),
`90f50cf` (paired Elo integration), and `cb63e11` (matched full/half evidence).

## Sampling and provenance

The experiment reads canonical Parquet directly and creates no derived
dataset. For every row it hashes the shard basename with 64-bit FNV-1a, mixes
the absolute zero-based row index with SplitMix64, and retains hash residues
`<4`, `<2`, or `<1` modulo four. The 0.25 subset is nested in 0.5, which is
nested in 1.0. Membership is independent of worker scheduling and global RNG.
This deliberately implements the user's corrected row-streaming design; it
does not infer game boundaries or claim within-game stratification.

The exact startup snapshot is [parquet-files.txt](parquet-files.txt): 497
sorted shard basenames, SHA-256
`ef7aa839e1ec2c01780123c42de086243995ec4cf9a7a0af3a0c01ffb7b3b595`.
Later atomic corpus appends could not enter these runs. The footer and hash
audit found:

| Retention | Accepted rows | Batch-usable rows |
| ---: | ---: | ---: |
| 1.0 | 4,074,985,928 | 4,058,644,480 |
| 0.5 | 2,037,509,235 | 2,021,195,776 |
| 0.25 | 1,018,779,501 | 1,003,094,016 |

Canonical `/parquet`, source manifests, and training configuration files were
not modified by this experiment. Checkpoints, exports, and evaluation outputs
use isolated `/artifacts` names.

## Matched training protocol

The cost planner selected the strongest measured family/configuration in the
original planning envelope: MoE64A2 d512x8 with Transformer Engine MXFP8. All
three launch summaries matched on model family, width, depth, optimizer/loss,
seed, batch, accepted samples, steps, and training FLOPs. Only retention and
output names differed.

| Quantity | Matched value |
| --- | ---: |
| Seed | 1 |
| Batch size | 65,536 |
| Steps | 15,000 |
| Accepted samples | 983,040,000 |
| Training ratio | 0.0468065454273x |
| Learning rate | 0.00037 |
| FLOPs/sample | 181,555,318 |
| Training FLOPs | 1.7847613980672e17 |
| Loader threads / prefetch | 8 / 2 per worker |

The lower-retention arms scanned approximately 1.97B and 3.93B underlying
rows to assemble the same accepted sample count as the full-retention arm; no
rows were repeated. The quarter arm's full run exceeded the original planning
estimate because scanning eventually became input-bound. The user explicitly
authorized completing it after clarifying that `$1.50` was not a strict cap.

Representative command (substitute `RATE` and the matching output name):

```sh
uv run train-modal --config configs/moe64a2.py --d-model 512 \
  --training-ratio 0.04680654542731256 --steps 15000 \
  --dataloader-threads 8 --dataloader-prefetch-per-thread 2 \
  --data-retention-rate RATE \
  --parquet-manifest experiments/2026-08-08.01-position-subsampling/parquet-files.txt \
  --wandb-name position-subsampling-rRATE
```

## Training results

Costs are rate-derived realized costs, not rounded Modal invoice line items:
B200 `$0.001736/s` plus eight CPU cores at `$0.0000131/core-s`, or
`$0.0018408/s`. Runtime is the W&B run duration, consistently applied to all
arms.

| Retention | W&B | Runtime | Cost | EMA task loss | Final policy | Final value | Final moves-left | EMA policy top-1 | Spikes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | [fyej1izp](https://wandb.ai/maxence-frenette/uncategorized/runs/fyej1izp) | 676.213 s | $1.244772 | 2.865781 | 2.026412 | 0.691026 | 0.147160 | 0.507204 | 0 |
| 0.5 | [4fe9jk45](https://wandb.ai/maxence-frenette/uncategorized/runs/4fe9jk45) | 663.502 s | $1.221374 | 2.848112 | 2.036135 | 0.678729 | 0.145244 | 0.512639 | 0 |
| 0.25 | [qhw06zrh](https://wandb.ai/maxence-frenette/uncategorized/runs/qhw06zrh) | 1,007.769 s | $1.855101 | 2.837690 | 2.031988 | 0.669298 | 0.140740 | 0.515694 | 0 |

Final task losses were 2.864599, 2.860108, and 2.842026 for retention 1.0,
0.5, and 0.25. Final policy top-1 was 0.512161, 0.513901, and 0.515823;
value-Q MSE was 0.031954, 0.031193, and 0.030953; moves-left MAE was
9.0021, 8.8788, and 8.6445. Overall accepted-sample throughput was 1.551M,
1.542M, and 1.007M samples/s. The final quarter interval fell to 0.710M
samples/s, explaining its higher runtime and cost. All runs had zero detected
loss spikes.

`compare-run` reported `EG_flops` of 5.575x, 8.004x, and 9.981x in retention
order. These trend comparisons are supporting diagnostics only; no best-run
file was changed.

Final checkpoints and exports:

| Retention | Checkpoint | lc0 export |
| ---: | --- | --- |
| 1.0 | `/artifacts/checkpoints/position-subsampling-r1-8t-final.pt` | `/artifacts/models/position-subsampling-r1-8t-final.safetensors` |
| 0.5 | `/artifacts/checkpoints/position-subsampling-r0.5-8t-final.pt` | `/artifacts/models/position-subsampling-r0.5-8t-final.safetensors` |
| 0.25 | `/artifacts/checkpoints/position-subsampling-r0.25-8t-full-final.pt` | `/artifacts/models/position-subsampling-r0.25-8t-full-final.safetensors` |

Earlier single-thread and prematurely stopped quarter attempts are invalid and
were not used in any comparison. Their retained evidence remains in the git
history and W&B rather than being overwritten.

## Searched lc0 tournament

The retained [tournament.toml](tournament.toml) specifies 800 visits per move,
three 128-game matchups on RTX PRO 6000, 64 parallel games, deterministic
temperature/noise settings, and the repository's two-move opening book. Every
opening was mirrored with colors swapped. This is 384 searched games / 192
paired openings in a complete three-candidate round robin; it is not
policy-only play.

The ce4 exports use lc0's native backend directly. A small evaluator fix was
required because the previous searched command wrapped the native ce4 backend
in legacy protobuf `multiplexing`, which attempted to parse Safetensors as a
protobuf network. A 16-game end-to-end probe then completed in 43.898 s. Its
conservative extrapolation was 1,053.6 s and `$0.915`, below the `$1` launch
gate. The actual tournament used 507.705 GPU-seconds. At RTX PRO 6000
`$0.000842/s` plus a conservative two CPU cores, its rate-derived cost was
`$0.440790`.

Command:

```sh
uv run eval-tournament-modal \
  --config experiments/2026-08-08.01-position-subsampling/tournament.toml \
  --output experiments/2026-08-08.01-position-subsampling/tournament-results.json
```

Direct matchup WDL is from the first model's perspective:

| Matchup | W-D-L | Runtime | Throughput | Pentanomial |
| --- | ---: | ---: | ---: | --- |
| 1.0 vs 0.5 | 31-46-51 | 166.624 s | 0.7682 games/s | [6, 30, 13, 8, 7] |
| 0.5 vs 0.25 | 43-40-45 | 156.763 s | 0.8165 games/s | [6, 15, 23, 15, 5] |
| 1.0 vs 0.25 | 39-40-49 | 184.318 s | 0.6945 games/s | [10, 14, 22, 12, 6] |

Aggregate searched-play throughput was 0.7563 games/s. This is the evaluator's
end-to-end inference/search throughput; the experiment did not substitute a
policy-only or synthetic backend benchmark.

The connected Bradley-Terry score quasi-MLE treats draws as 0.5 and reports
two-sided 95% Wald intervals with a CR1 sandwich covariance clustered by the
64 reused indexed mirrored openings. All 192 ordered pair scores are retained
in [tournament-results.json](tournament-results.json).

| Rank | Retention | Centered Elo | 95% CI |
| ---: | ---: | ---: | ---: |
| 1 | 0.5 | +16.36 | [-6.07, +38.80] |
| 2 | 0.25 | +10.91 | [-15.72, +37.54] |
| 3 | 1.0 | -27.27 | [-49.93, -4.61] |

The 0.5 and 0.25 intervals overlap heavily, and their direct match was nearly
even. Both reduced value and moves-left training losses and beat 1.0 in their
direct searched matchups. This supports the value-diversity hypothesis, while
not resolving whether 0.5 or 0.25 is optimal.

## Recommendation

Retain the deterministic sampler and 0.5 retention as the leading candidate
for review. It delivered almost all of quarter retention's training and search
gain without the quarter arm's 52% runtime/cost increase, and its tournament
point estimate was highest. Reject 1.0 as the preferred treatment for this
matched allocation. Do not promote 0.5, edit canonical data/configs, or merge
this branch until the user reviews the evidence. If a follow-up is desired,
spend it on additional paired 0.5-vs-0.25 searched games rather than another
training sweep.
