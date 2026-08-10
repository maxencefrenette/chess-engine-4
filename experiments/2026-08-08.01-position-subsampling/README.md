# Position subsampling for value diversity

## Outcome

Fresh random row sampling improved both task loss and searched play at matched
accepted samples and training FLOPs. The ranking was 0.25 > 0.5 > 1.0. At 800
visits, quarter beat half by `+32.17 Elo [19.68, 44.66]`, while half beat full
by `+42.74 [31.47, 54.02]`. Promote 0.25 as the canonical training sampling
rate.

The quarter run recorded two isolated automatic spike flags around steps 2,450
and 7,010, then immediately returned to trend. Sampling rate was the only
changed variable and learning rate was fixed, so these flags do not disqualify
the result.

Task: `01kzg0rdwf`

Branch: `codex/position-subsampling-01kzg0rdwf`

Base commit: `316c327da2612f5b71e9776a9ae8d3fc773c216d`

Random-sampling implementation: `4e20293b`

## Protocol

The loader streamed the canonical Parquet directory and selected a fresh
random subset for every launch. It created no derived dataset and did not
modify canonical data. All three arms used the same 497-shard startup snapshot,
MoE64A2 d512x8 model, optimizer and loss recipe, seed 1, learning rate
`0.00037`, batch 65,536, 15,000 steps, and 983,040,000 accepted samples.
Training compute was `1.7847613980672e17` FLOPs per arm. The production loader
used eight threads with two prefetched batches per worker.

```sh
uv run train-modal --config configs/moe64a2.py --d-model 512 \
  --training-ratio 0.04680654542731256 --steps 15000 \
  --dataloader-threads 8 --dataloader-prefetch-per-thread 2 \
  --sampling-rate RATE \
  --wandb-name position-subsampling-random-rRATE
```

## Training results

Costs use B200 `$0.001736/s` plus eight CPU cores at
`$0.0000131/core-s`. Each arm consumed exactly 983,040,000 accepted rows.

| Sampling rate | W&B | Runtime | Cost | EMA task loss | Policy loss | Value loss | Moves-left loss | EMA policy top-1 | Spikes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | [xqwf95gr](https://wandb.ai/maxence-frenette/uncategorized/runs/xqwf95gr) | 651.804 s | $1.199841 | 2.865800 | 2.026427 | 0.691094 | 0.147131 | 0.507592 | 0 |
| 0.5 | [2o60pa22](https://wandb.ai/maxence-frenette/uncategorized/runs/2o60pa22) | 658.386 s | $1.211957 | 2.847041 | 2.037432 | 0.666627 | 0.140095 | 0.512505 | 0 |
| 0.25 | [qhqb7a9c](https://wandb.ai/maxence-frenette/uncategorized/runs/qhqb7a9c) | 691.347 s | $1.272631 | 2.835924 | 2.024474 | 0.674327 | 0.140235 | 0.515167 | 2 |

Total training cost was `$3.684429`. Realized costs were within 6.1%. Quarter
had the best task loss and policy top-1; half had the best final value-Q MSE
(`0.030509`) and moves-left MAE (`8.6185`). Before promotion, `compare-run`
reported quarter at `10.368x EG_flops`, `BEATS TREND`, and `PROMOTE`.

The retained checkpoints and lc0 exports are:

| Sampling rate | Checkpoint | lc0 export |
| ---: | --- | --- |
| 1.0 | `/artifacts/checkpoints/position-subsampling-random-r1-final.pt` | `/artifacts/models/position-subsampling-random-r1-final.safetensors` |
| 0.5 | `/artifacts/checkpoints/position-subsampling-random-r0.5-final.pt` | `/artifacts/models/position-subsampling-random-r0.5-final.safetensors` |
| 0.25 | `/artifacts/checkpoints/position-subsampling-random-r0.25-final.pt` | `/artifacts/models/position-subsampling-random-r0.25-final.safetensors` |

## Searched tournament

The three exports played a connected round robin at 800 visits on RTX PRO
6000. The tournament used `UHO_Lichess_4852_v1`, mirrored every opening, ran
64 games in parallel, and allowed inference batches up to 256 positions. The
book SHA-256 was
`dd1b5de3efadd40f9ecaa7392c69545de6f2e44ba9213362e4c6723ce7db803c`.

| Matchup | Opening pairs | W-D-L | Runtime | Pentanomial |
| --- | ---: | ---: | ---: | --- |
| 1.0 vs 0.5 | 928 | 611-369-876 | 968.051 s | [174, 181, 387, 108, 78] |
| 0.5 vs 0.25 | 640 | 425-274-581 | 559.210 s | [110, 132, 261, 78, 59] |
| 1.0 vs 0.25 | 640 | 397-252-631 | 551.678 s | [132, 133, 259, 69, 47] |

The aggregate 2,208-pair fit is retained in
[tournament-random-rerun-combined-results.json](tournament-random-rerun-combined-results.json).

| Rank | Sampling rate | Centered Elo | 95% CI |
| ---: | ---: | ---: | ---: |
| 1 | 0.25 | +35.70 | [+28.21, +43.18] |
| 2 | 0.5 | +3.52 | [-3.20, +10.24] |
| 3 | 1.0 | -39.22 | [-46.03, -32.41] |

Pairwise contrasts were 0.25 minus 0.5 = `+32.17 [19.68, 44.66]`, 0.5
minus 1.0 = `+42.74 [31.47, 54.02]`, and 0.25 minus 1.0 =
`+74.92 [62.29, 87.55]`. The 4,416 games took 2,078.939 GPU-seconds and cost
`$1.804934` at RTX PRO 6000 `$0.000842/s` plus two CPU cores. Completed-play
throughput was 2.124 games/s.

```sh
uv run eval-tournament-modal --config \
  experiments/2026-08-08.01-position-subsampling/tournament-random-rerun.toml \
  --output experiments/2026-08-08.01-position-subsampling/tournament-random-rerun-results.json
uv run eval-tournament-modal --config \
  experiments/2026-08-08.01-position-subsampling/tournament-random-rerun-half-quarter.toml \
  --output experiments/2026-08-08.01-position-subsampling/tournament-random-rerun-half-quarter-results.json
uv run eval-tournament-modal --config \
  experiments/2026-08-08.01-position-subsampling/tournament-random-rerun-full-quarter.toml \
  --output experiments/2026-08-08.01-position-subsampling/tournament-random-rerun-full-quarter-results.json
uv run python experiments/2026-08-08.01-position-subsampling/analyze_uho.py \
  --random-rerun
```

## Recommendation

Use fresh random sampling at 0.25 for canonical training. It improves the
matched task objective without a policy regression and decisively leads both
alternatives in searched play.
