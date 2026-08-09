# Position subsampling for value diversity

## Outcome

Three matched `moe64a2 d512` treatments completed at retention rates 1.0, 0.5,
and 0.25. Each consumed exactly 983,040,000 accepted rows in 15,000 optimizer
steps. Half and quarter retention both improved training loss and searched-play
strength relative to full retention. The primary randomized-UHO tournament
resolves the ranking: quarter retention leads half by 25.35 Elo with 95% CI
`[13.66, 37.05]`, and half leads full by 35.58 Elo with 95% CI
`[23.89, 47.28]`. Retain 0.25 as the leading candidate; do not promote it to a
canonical configuration until the user reviews this report.

Task: `01kzg0rdwf`  
Branch: `codex/position-subsampling-01kzg0rdwf`  
Base commit: `316c327da2612f5b71e9776a9ae8d3fc773c216d`

Relevant implementation commits are `3bbb430` (deterministic streaming
retention), `13dcaf0` (initial-run evidence), `7a51a48` (eight-thread probe),
`90f50cf` (paired Elo integration), and `cb63e11` (matched full/half evidence).
Completion commit `8d327b3d` retains the evaluator fix, tournament protocol,
raw paired results, final report, and task disposition.

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

## Recommendation

Retain the deterministic sampler and 0.25 retention as the leading candidate
for review. It has the best task/value/moves-left losses and now beats 0.5 by a
statistically resolved 25.35 Elo in the primary searched tournament. Its cost
is the tradeoff: quarter-retention training took 52% longer and cost 52% more
than half retention because of additional Parquet scanning. Reject 1.0 as the
preferred treatment for this matched allocation. Do not promote 0.25, edit
canonical training data/configs, or merge this branch until the user reviews
the evidence.
