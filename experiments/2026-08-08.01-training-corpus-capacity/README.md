# Training corpus expansion and capacity audit

## Outcome

The canonical LCZero Parquet corpus expanded from `480` shards and
`3,949,735,220` rows to `1,203` shards and `8,020,779,820` rows. The final
Parquet payload is `696,169,217,477` bytes (`648.358 GiB`), or `86.795702`
bytes per row. This adds `723` shards, `4,071,044,600` usable positions, and
`353,448,459,341` Parquet bytes without replacing an existing shard.

The approved cost-limited candidate requires `1,305,804,800` training rows.
The separate quarter-rate treatment therefore needs at least
`5,223,219,200` raw positions; the final canonical corpus has
`8,020,779,820`, leaving `2,797,560,620` raw positions of capacity above that
requirement. This report does not launch or evaluate the separate subsampling
experiment.

After verified source cleanup, the training-data Volume uses
`696,169,464,367` bytes and the artifact Volume uses `48,844,737,130` bytes.
Combined usage is `745,014,201,497` bytes (`693.849 GiB`), leaving
`221,353,440,103` bytes (`206.151 GiB`) below the `900 GiB`
(`966,367,641,600` byte) operational ceiling. There are no retained `.tar`,
`.tmp`, or `.parquet.partial` files.

Modal's current pricing page says the included storage allowance is `1 TiB / mo`.
The allowance is therefore binary TiB (`1,099,511,627,776` bytes), not decimal
TB, and applies across the workspace rather than only to training data. The
`900 GiB` operational ceiling retains `124 GiB` below that paid boundary for
metadata, staging, and artifacts. Modal documents that Volume usage is
snapshotted daily and deletions can remain billable for up to four days.

Official references:

- <https://modal.com/pricing>
- <https://modal.com/docs/guide/volumes>

## Acquisition and provenance

The upstream test80 listing exposed `9,181` nontrivial archives at inventory
time. The `1,454,080`-byte
`training-run1-test80-20240428-1817.tar` was treated as anomalous and excluded
by the `100 MiB` minimum source-size rule.

The expansion used `723` unique upstream archives from
`training-run1-test80-20240428-1417.tar` through
`training-run1-test80-20240528-1717.tar`, totaling `836,396,574,720` source
bytes. Every archive matched its advertised size and SHA-256 before conversion.
The complete machine-readable provenance index is `sources.toml`; the live
per-source JSON manifests remain under `/source-manifests`, and `14` exact
selection manifests remain under `/source-manifests/sync-runs`.

The provenance audit found `723` unique names, `723` unique SHA-256 values, and
all `723` corresponding Parquet outputs in the canonical prefix. Candidate
selection excluded every already-converted source name, every batch committed
its exact selection before download, and postflight checks rejected any
complete source outside the initial-plus-selected set.

The original first `17` expansion archives were fully decoded before cleanup:
they contained `1,143,453` games and `125,250,708` positions with zero repeated
game IDs. Later archives were intentionally deleted after verified conversion,
as directed by the user, and Parquet does not retain game IDs or game
boundaries. Consequently, exact total games and a corpus-wide game-ID duplicate
count cannot be reconstructed without reacquiring source data. Archive-level
identity is fully audited; game-level uniqueness beyond the first `17` sources
is an explicit limitation rather than an inferred claim.

## Conversion and correctness

Acquisition was serial and bounded. Before every batch, the synchronizer read
both Modal Volumes and reserved the selected source bytes plus a full
source-sized output allocation. It downloaded through `.tmp`, verified exact
upstream size and SHA-256, and committed a persistent sync-run manifest before
the first source. A stopped Modal app was safely reattached to the same exact
manifest with `--sync-run-id`; no new selection was made.

Conversion wrote `.parquet.partial` and atomically renamed only completed
outputs. It never replaced an existing Parquet shard. Every added shard passed:

- native LCZero-to-Parquet conversion and row counting;
- four conversion-time source/output verification batches;
- a separate fresh native-loader source/output comparison before source
  deletion; and
- the final all-shard footer audit.

The exact source/output comparisons covered packed planes, scalar planes,
compact policy indices, FP16 policy bits, and root Q/D/M. The final footer audit
confirmed `1,203` readable shards, `8,020,779,820` rows, and
`696,169,217,477` Parquet bytes.

A bounded production-loader audit decoded one 256-row batch from every shard
(`307,968` sampled rows). It found zero non-finite targets, zero Q values
outside `[-1, 1]`, zero D values outside `[0, 1]`, zero invalid implied WDL
triples, and zero negative moves-left values. Sampled distributions were:

| Target | Mean | Std. dev. | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| root Q | -0.034312 | 0.414529 | -1.000000 | 1.000000 |
| root D | 0.568073 | 0.264675 | 0.000000 | 0.999998 |
| root M | 142.011806 | 50.502153 | 1.000700 | 243.085205 |

This audit also confirms final-schema compatibility through the production Rust
Parquet loader on all `1,203` shards. Loader throughput was not rebenchmarked by
launching training: the unchanged loader previously measured `1.05M`
positions/s in the loader-level benchmark and completed the canonical B200
throughput profiles recorded in `experiments/2026-08-04.04-parquet-data` and
`experiments/throughput-moe64a2.toml`.

## Operational notes

One cleanup call removed `12` of `16` already-verified sources before Modal
returned `No such file or directory`. Read-only reconciliation found exactly
the remaining four expected sources; all four passed a fresh comparison again
and were then removed. No Parquet data was deleted or rolled back.

Acquisition stopped after finishing the already-selected final `96`-source
batch, as directed by the user. Reaching `900 GiB` of training data alone is
incompatible with the all-workspace `900 GiB` combined ceiling while any
artifact storage remains; no attempt was made to cross the combined ceiling,
and no completed Parquet work was discarded.

W&B URL and `EG_flops` are not applicable because this was a data operation and
no training run was launched.

## Commands

```sh
uv run python scripts/sync_modal_training_data.py \
  --start-day 20240428 --file-count BATCH --download-concurrency 1
uv run python scripts/sync_modal_training_data.py \
  --start-day 20240428 --file-count 32 --download-concurrency 1 \
  --sync-run-id 4e600f462eab4cbdb014f12b0d46babb
uv run convert-data-modal --all --verify-batches 4
uv run verify-data-modal --batches 1 --delete-sources
uv run audit-data-modal
uv run python scripts/audit_modal_parquet_targets.py
```
