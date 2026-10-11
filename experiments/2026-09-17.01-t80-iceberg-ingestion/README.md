# Full T80 Iceberg ingestion

Recorded retrospectively on 2026-10-11 from the ingestion status/log and the
2026-09-17 completion audit. This was a data conversion, not a model experiment;
there is no W&B run, EG_flops measurement, or model promotion.

## Scope and command

The frozen T80 inventory contained 9,958 source archives totaling
8,992,666,368,000 advertised bytes. The converter retained packed inputs,
compact binary policy targets, `root_*`, `best_*`, `result_q/d`, `plies_left`,
`num_pieces`, and `source_archive`. The Iceberg table uses
`truncate[4](num_pieces)` partitions.

```sh
uv run ingest-iceberg /data/chess/t80/source /data/chess/t80/iceberg \
  --table training.t80 --workers 4 --commit-batch-size 16
```

The completed table is `training.t80`, with the local SQLite catalog at
`/data/chess/t80/iceberg/catalog.db`. Conversion was resumed across interruptions;
completed partition outputs were recovered before committing them to Iceberg.

## Results and audit

| Metric | Result |
| --- | ---: |
| Recorded source archives | 9,958 |
| Training positions | 43,802,607,232 |
| Parquet data files | 82,629 |
| Compressed data bytes | 4,235,088,583,781 |
| Output / source bytes | 47.1% |
| Final ingestion snapshot | 4235643209546647011 |
| Completion time | 2026-09-17 02:10:32 UTC |

The completion audit matched the current snapshot's file references against
physical files: no duplicate references, missing files, unreferenced Parquet
files, or partial files remained. Manifest row and byte totals matched the
ingestion status, and the schema and partition spec were checked. This was a
metadata/file audit, not a full reread of every converted row.

The log records 146 omitted corrupt gzip members across 16 source archives.
All other members in those archives were processed. Fresh downloads of two
failing archives matched the originals byte for byte, supporting upstream
corruption for those archives; this comparison was not performed for every
omitted member. The omitted members cannot be assigned a reliable lost-row
count from the completed output.

After the audit, all 9,958 source tar files and 11 quarantined duplicate archives
were deleted. The Iceberg warehouse, inventory, ledger, and logs were retained.
The durable operational records are `ingest.status`, `ingest.log`, and
`ingested-archives.txt` under the warehouse.

## Remaining integration

This corpus is distinct from the older 8,020,779,820-position Modal Parquet
corpus. The current training reader expects the older column types, so the
Iceberg data cannot yet be passed directly to training. Reader integration,
reproducible selection/mixing, and verified Modal upload remain tracked in
[the staging task](../../tasks/data/01kzmczgtt-stage-large-training-datasets-from-the-desktop.md).
