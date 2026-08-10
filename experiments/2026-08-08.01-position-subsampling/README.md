# Position subsampling for value diversity

## Outcome

Three matched `moe64a2 d512` treatments completed at retention rates 1.0, 0.5,
and 0.25, then the entire experiment was replicated with fresh random row
membership. Every run consumed exactly 983,040,000 accepted rows in 15,000
optimizer steps. Both experiments rank searched strength 0.25 > 0.5 > 1.0.
In the fresh-random replication, quarter leads half by 32.17 Elo with 95% CI
`[19.68, 44.66]`, and half leads full by 42.74 Elo with 95% CI
`[31.47, 54.02]`. The diversity result holds. Retain random subsampling for
review, but do not canonically promote the quarter run because it recorded two
automatic loss-spike flags.

Task: `01kzg0rdwf`  
Branch: `codex/position-subsampling-01kzg0rdwf`  
Base commit: `316c327da2612f5b71e9776a9ae8d3fc773c216d`

Relevant implementation commits are `3bbb430` (deterministic streaming
retention), `4e20293b` (fresh random membership), `13dcaf0` (initial-run evidence), `7a51a48` (eight-thread probe),
`90f50cf` (paired Elo integration), and `cb63e11` (matched full/half evidence).
Completion commit `8d327b3d` retains the evaluator fix, tournament protocol,
raw paired results, final report, and task disposition. UHO migration,
randomized-pair provenance, native lc0 batching, and the expanded tournament
evidence are retained in `fd04acaf`.

## Sampling and provenance

The experiment reads canonical Parquet directly and creates no derived
dataset. For every row it hashes the shard basename with 64-bit FNV-1a, mixes
the absolute zero-based row index with SplitMix64, and retains hash residues
`<4`, `<2`, or `<1` modulo four. The 0.25 subset is nested in 0.5, which is
nested in 1.0. Membership is independent of worker scheduling and global RNG.
This deliberately implements the user's corrected row-streaming design; it
does not infer game boundaries or claim within-game stratification.

The loader resolved 497 Parquet shards from the canonical data directory when
these runs constructed their datasets. Later atomic corpus appends could not
enter an already-running iterator. The footer audit found:

| Retention | Accepted rows | Batch-usable rows |
| ---: | ---: | ---: |
| 1.0 | 4,074,985,928 | 4,058,644,480 |
| 0.5 | 2,037,509,235 | 2,021,195,776 |
| 0.25 | 1,018,779,501 | 1,003,094,016 |

Canonical `/parquet` and training configuration files were not modified by
this experiment. Checkpoints, exports, and evaluation outputs use isolated
`/artifacts` names.

After the initial experiment, the user clarified that each launch should draw
a fresh random subset rather than reuse fixed row membership. The current
loader therefore mixes the stable shard/row identity with a fresh per-iterator
seed. The seed is shared across workers, so membership is independent of
prefetch scheduling, and is printed in the launch log for provenance. There is
still no derived dataset or mutation of canonical Parquet.

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

## Fresh-random replication

The complete matched training experiment was repeated after changing only row
membership from fixed to fresh random samples. The same 497-shard corpus state,
model recipe, seed, batch, steps, accepted samples, optimizer, and training
FLOPs were retained. The logged sampling seeds were
`16744882689526351630`, `11635758382966936071`, and
`9229423701945907918` in retention order.

| Retention | W&B | Runtime | Cost | EMA task loss | Final policy | Final value | Final moves-left | EMA policy top-1 | Spikes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | [xqwf95gr](https://wandb.ai/maxence-frenette/uncategorized/runs/xqwf95gr) | 651.804 s | $1.199841 | 2.865800 | 2.026427 | 0.691094 | 0.147131 | 0.507592 | 0 |
| 0.5 | [2o60pa22](https://wandb.ai/maxence-frenette/uncategorized/runs/2o60pa22) | 658.386 s | $1.211957 | 2.847041 | 2.037432 | 0.666627 | 0.140095 | 0.512505 | 0 |
| 0.25 | [qhqb7a9c](https://wandb.ai/maxence-frenette/uncategorized/runs/qhqb7a9c) | 691.347 s | $1.272631 | 2.835924 | 2.024474 | 0.674327 | 0.140235 | 0.515167 | 2 |

Every arm again consumed exactly 983,040,000 accepted rows in 15,000 steps.
Fresh random membership reproduced the task-loss ordering and avoided the
original quarter-rate loader slowdown: realized costs were within 6.1% of one
another. Half retention had the best final value-Q MSE (`0.030509`) and
moves-left MAE (`8.6185`); quarter retained the best EMA task loss and policy
top-1. Full and half had zero detected loss spikes. Quarter recorded two
one-off spike flags around logged steps 2,450 and 7,010; its subsequent losses
and gradients returned immediately to trend, but repository methodology still
disqualifies that run from canonical promotion.

`compare-run` reports `EG_flops` 5.573x for full and 8.186x for half, with
`BEATS TREND` / `PROMOTE` diagnostic verdicts. It correctly refuses to assign
an `EG_flops` or promotion verdict to quarter because of the two spike flags.

Fresh-random checkpoints and exports:

| Retention | Checkpoint | lc0 export |
| ---: | --- | --- |
| 1.0 | `/artifacts/checkpoints/position-subsampling-random-r1-final.pt` | `/artifacts/models/position-subsampling-random-r1-final.safetensors` |
| 0.5 | `/artifacts/checkpoints/position-subsampling-random-r0.5-final.pt` | `/artifacts/models/position-subsampling-random-r0.5-final.safetensors` |
| 0.25 | `/artifacts/checkpoints/position-subsampling-random-r0.25-final.pt` | `/artifacts/models/position-subsampling-random-r0.25-final.safetensors` |

Representative replication commands (substitute `RATE` in all three names):

```sh
uv run train-modal --config configs/moe64a2.py --d-model 512 \
  --training-ratio 0.04680654542731256 --steps 15000 \
  --dataloader-threads 8 --dataloader-prefetch-per-thread 2 \
  --data-retention-rate RATE \
  --wandb-name position-subsampling-random-rRATE

uv run export-model \
  /artifacts/checkpoints/position-subsampling-random-rRATE-final.pt \
  --output artifacts/models/position-subsampling-random-rRATE-final.safetensors \
  --remote-only
```

Final checkpoints and exports:

| Retention | Checkpoint | lc0 export |
| ---: | --- | --- |
| 1.0 | `/artifacts/checkpoints/position-subsampling-r1-8t-final.pt` | `/artifacts/models/position-subsampling-r1-8t-final.safetensors` |
| 0.5 | `/artifacts/checkpoints/position-subsampling-r0.5-8t-final.pt` | `/artifacts/models/position-subsampling-r0.5-8t-final.safetensors` |
| 0.25 | `/artifacts/checkpoints/position-subsampling-r0.25-8t-full-final.pt` | `/artifacts/models/position-subsampling-r0.25-8t-full-final.safetensors` |

Earlier single-thread and prematurely stopped quarter attempts are invalid and
were not used in any comparison. Their retained evidence remains in the git
history and W&B rather than being overwritten.

## Primary searched lc0 tournament

The initial 384-game result in [tournament-results.json](tournament-results.json)
used the first 64 sequential entries of `noob_2moves`. A later audit found that
all 64 were concentrated on white e-pawn openings. It remains historical
evidence but is not combined with the primary result below.

All current Elo and tournament defaults now use a pinned sample of
`UHO_Lichess_4852_v1`. [uho-book-manifest.json](uho-book-manifest.json) records
official-stockfish/books commit `65815ccd`, all source/sample hashes, 2,632,036
source positions, and the uniform reservoir sample of 65,536 positions. lc0
reproducibly shuffles that sample with seed 1. Every selected opening is played
as an adjacent mirrored color pair; explicit offsets prevent accidental reuse.

The main [tournament-uho-extended.toml](tournament-uho-extended.toml) runs a
complete 4,416-game / 2,208-pair round robin at 800 visits. All three matchups
share randomized opening offsets 32–767, allowing the paired estimator to
cluster the same position across matchups. After the connected fit, the
information-gain scheduler selected an additional 1.0-vs-0.25 match. The
[extra configuration](tournament-uho-extra.toml) contributes 1,136 games / 568
fresh pairs at offsets 768–1,335.

```sh
uv run eval-tournament-modal \
  --config experiments/2026-08-08.01-position-subsampling/tournament-uho-extended.toml \
  --output experiments/2026-08-08.01-position-subsampling/tournament-uho-extended-results.json

uv run eval-tournament-modal \
  --config experiments/2026-08-08.01-position-subsampling/tournament-uho-extra.toml \
  --output experiments/2026-08-08.01-position-subsampling/tournament-uho-extra-results.json

uv run python experiments/2026-08-08.01-position-subsampling/analyze_uho.py
```

The native ce4 backend now queues concurrent lc0 search computations, waits up
to 200 microseconds, and coalesces them into inference batches up to 256. A
64-game UHO probe averaged batches 187.5 and 192.4. Across the paid matches,
model/match averages ranged from 176.8 to 241.7 and every backend reached batch
256. End-to-end searched-play throughput was 2.473 games/s, versus 0.756 for
the unbatched historical tournament.

Direct WDL is from the first model's perspective:

| Matchup | Opening pairs | W-D-L | Runtime | Pentanomial |
| --- | ---: | ---: | ---: | --- |
| 1.0 vs 0.5 | 736 | 497-311-664 | 581.835 s | [129, 149, 284, 108, 66] |
| 0.5 vs 0.25 | 736 | 515-318-639 | 647.725 s | [117, 142, 296, 110, 71] |
| 1.0 vs 0.25 | 736 | 471-310-691 | 574.947 s | [142, 151, 292, 87, 64] |
| 1.0 vs 0.25, fresh extension | 568 | 347-226-563 | 440.151 s | [124, 108, 237, 58, 41] |

All 2,776 ordered pair scores and stable opening identities are retained in
the two raw result files. [tournament-uho-combined-results.json](tournament-uho-combined-results.json)
contains the combined 1,304-cluster fit. The Bradley-Terry quasi-MLE treats
draws as 0.5 and uses two-sided 95% Wald intervals with a CR1 sandwich
covariance clustered by opening identity.

| Rank | Retention | Centered Elo | 95% CI |
| ---: | ---: | ---: | ---: |
| 1 | 0.25 | +28.76 | [+22.57, +34.95] |
| 2 | 0.5 | +3.41 | [-3.61, +10.43] |
| 3 | 1.0 | -32.17 | [-38.36, -25.98] |

The correlated pairwise contrasts, which are the correct significance tests,
are 0.25 minus 0.5 = `+25.35 Elo [13.66, 37.05]`, 0.5 minus 1.0 =
`+35.58 [23.89, 47.28]`, and 0.25 minus 1.0 =
`+60.94 [50.73, 71.14]`. The randomized-UHO tournament therefore resolves all
three treatments rather than merely narrowing their marginal intervals.

Tournament runtime was 2,244.657 GPU-seconds. At RTX PRO 6000 `$0.000842/s`
plus a conservative two CPU cores, its rate-derived cost was `$1.948812`.
Including the 33.089-second batching probe gives `$1.977539`, approximately
the requested `$2` allocation.

## Fresh-random searched replication

The three fresh-random exports were compared at 800 visits with the pinned UHO
book. Every randomly shuffled opening was played as an adjacent mirrored pair.
The initial plan used 1,856 games per matchup, but its first measured match
took 968.051 seconds and projected to `$2.52`. It was stopped after wave one;
the two remaining matchups were resized to 1,280 games each, projecting the
complete connected comparison to `$1.9997`. No completed evidence was dropped.

```sh
uv run eval-tournament-modal \
  --config experiments/2026-08-08.01-position-subsampling/tournament-random-rerun.toml \
  --output experiments/2026-08-08.01-position-subsampling/tournament-random-rerun-results.json

uv run eval-tournament-modal \
  --config experiments/2026-08-08.01-position-subsampling/tournament-random-rerun-half-quarter.toml \
  --output experiments/2026-08-08.01-position-subsampling/tournament-random-rerun-half-quarter-results.json

uv run eval-tournament-modal \
  --config experiments/2026-08-08.01-position-subsampling/tournament-random-rerun-full-quarter.toml \
  --output experiments/2026-08-08.01-position-subsampling/tournament-random-rerun-full-quarter-results.json

uv run python experiments/2026-08-08.01-position-subsampling/analyze_uho.py \
  --random-rerun
```

The launch-generated opening seeds were `250116766`, `651675137`, and
`1099564731`. The three raw result files retain all 2,208 ordered pair scores
and opening identities. Direct WDL is from the first model's perspective:

| Matchup | Opening pairs | W-D-L | Runtime | Pentanomial |
| --- | ---: | ---: | ---: | --- |
| 1.0 vs 0.5 | 928 | 611-369-876 | 968.051 s | [174, 181, 387, 108, 78] |
| 0.5 vs 0.25 | 640 | 425-274-581 | 559.210 s | [110, 132, 261, 78, 59] |
| 1.0 vs 0.25 | 640 | 397-252-631 | 551.678 s | [132, 133, 259, 69, 47] |

The connected clustered fit in
[tournament-random-rerun-combined-results.json](tournament-random-rerun-combined-results.json)
gives:

| Rank | Retention | Centered Elo | 95% CI |
| ---: | ---: | ---: | ---: |
| 1 | 0.25 | +35.70 | [+28.21, +43.18] |
| 2 | 0.5 | +3.52 | [-3.20, +10.24] |
| 3 | 1.0 | -39.22 | [-46.03, -32.41] |

The correlated contrasts are 0.25 minus 0.5 = `+32.17 Elo [19.68, 44.66]`,
0.5 minus 1.0 = `+42.74 [31.47, 54.02]`, and 0.25 minus 1.0 =
`+74.92 [62.29, 87.55]`. Thus fresh random membership reproduces and slightly
strengthens the original searched ranking.

The matches used 2,078.939 completed GPU-seconds and cost `$1.804934` at the
same RTX PRO 6000 plus two-CPU rate. This rate-derived figure excludes startup
and the few aborted seconds immediately after the oversized plan entered wave
two, so it is not a rounded Modal invoice line item. End-to-end completed-play
throughput was 2.124 games/s. Backend inference batches averaged 177.5-240.3
positions, every model reached batch 256, and per-model inference throughput
ranged from 57.3k to 70.5k positions/s.

## Recommendation

Retain fresh random subsampling: the loss and searched-play ranking replicated
decisively, and the rerun kept all three training costs within 6.1%. Quarter
retention is the strongest searched model, but its two automatic spike flags
make that particular run ineligible for canonical promotion. Half retention is
the clean zero-spike candidate and still beats full by `+42.74 Elo`. Reject 1.0
as the preferred treatment for this matched allocation. Do not change
canonical training configs or merge this branch until the user reviews the
evidence; obtain a clean quarter checkpoint before promoting 0.25.
