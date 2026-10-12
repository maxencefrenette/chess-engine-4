# Training data and maintenance

Build the local Rust loader and inspect the configured Parquet dataset:

```sh
uv run maturin develop --manifest-path crates/leela_loader/Cargo.toml --release
uv run inspect-data
```

Convert a local LCZero archive with:

```sh
uv run lc0-to-parquet training.tar training.parquet
```

There are currently two distinct corpus formats:

| Corpus | Metadata | Status |
| --- | --- | --- |
| Existing Modal Parquet corpus | `experiments/training-data.toml` | 8,020,779,820 positions; supported by the current training reader |
| Local T80 Iceberg table | `experiments/training-data-iceberg.toml` | 43,802,607,232 positions; ingestion complete, training integration pending |

The local Iceberg warehouse is `/data/chess/t80/iceberg`, with table `training.t80`
and a SQLite catalog at `catalog.db`. It is partitioned by
`truncate[4](num_pieces)` and retains source-archive provenance and root, best,
outcome, and moves-left targets. The original LCZero archives were deleted after
the completion audit. See the [ingestion report](../experiments/2026-09-17.01-t80-iceberg-ingestion/README.md)
for counts, compression, and omitted corrupt members.

The Iceberg files use different column types from the older Parquet format.
Pointing `train-local --data` at them is not yet supported. Remaining work includes
the reader, reproducible filtering/shuffling, and verified transfer to Modal.
The Iceberg metadata can already supply a planning capacity with
`--dataset experiments/training-data-iceberg.toml`; this does not stage the data
or make a proposed run launchable against Iceberg.
The original acquisition/conversion commands remain maintenance tools; completed
conversion does not imply Iceberg training support. Cloud data commands are in
[cloud.md](cloud.md#data-maintenance).
