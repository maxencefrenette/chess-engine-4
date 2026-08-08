# Training corpus capacity audit

## Outcome

Canonical corpus expansion is blocked by source-retention storage, not by the
availability of upstream test80 archives. No canonical Parquet shard was added
and `experiments/training-data.toml` remains unchanged.

Modal's current Volume pricing includes `1 TiB/month` across the workspace's
volumes. Before acquisition, the training volume used `342,720,758,136` bytes
and the artifact volume used `347,573,940,073` bytes, or `690,294,698,209`
bytes combined. This audit used a `900 GiB` (`966,367,641,600` byte) combined
operational ceiling, retaining `124 GiB` below the paid boundary.

The canonical Parquet footer audit independently confirmed `3,949,735,220`
rows across `480` shards in `342,720,758,136` bytes, or `86.770565` bytes per
row.

## Source inventory

The upstream test80 listing contained `9,181` nontrivial archives at inventory
time. Canonical source-name overlap accounts for `432` of the current shards;
the other `48` older canonical sources are no longer listed upstream. The
`1,454,080`-byte `training-run1-test80-20240428-1817.tar` was treated as
anomalous and excluded.

Seventeen new archives were retained at the training-volume root, from
`training-run1-test80-20240428-1417.tar` through
`training-run1-test80-20240429-0717.tar`, totaling `25,723,576,320` bytes. Each
completed download exactly matched its advertised upstream size and has a
SHA-256 sidecar under `/source-manifests`. The same provenance is retained in
`sources.toml` beside this report.

All 17 archives were fully decoded and validated. They contain `1,143,453`
games and `125,250,708` v6 positions, with valid tar/gzip members and record
boundaries and no repeated canonical game IDs. The first eight sources use
`12,050,606,080` bytes for `58,681,470` positions, an observed density of
`205.356241` compressed source bytes per raw position.

After acquisition stopped, live usage was `368,444,339,114` bytes in training
data and `347,573,940,073` bytes in artifacts, or `716,018,279,187` bytes
combined. This leaves `250,349,362,413` bytes below the 900 GiB operational
ceiling and `383,493,348,589` bytes below 1 TiB.

## Capacity blocker

The matched subsampling experiment selected a `moe64a2 d512` treatment with
`1,305,804,800` rows. Its quarter-rate treatment needs about
`5,223,219,200` raw positions.

The safety policy reserves one source-sized output allocation for every
retained but unconverted tar. From the pre-source baseline, the 900 GiB ceiling
therefore admits at most:

```text
floor((966,367,641,600 - 690,294,698,209) / 2)
= 138,036,471,695 source bytes
```

At the audited density, that supports approximately `672,180,554` raw
positions or `168,045,139` nominal quarter-rate rows, `7.7706x` short of the
required treatment. Fitting the requirement would need future archives to use
at most `26.427` compressed bytes per raw position, an unsupported `7.77x`
improvement over the audited source density. Acquisition and canonical
conversion therefore stopped rather than approaching the billing boundary or
guessing a different training target.

## Commands

```sh
uv run audit-data-modal
uv run python scripts/sync_modal_training_data.py \
  --start-day 20240428 --file-count 8 --dry-run
uv run python scripts/sync_modal_training_data.py \
  --start-day 20240428 --file-count 8
uv run audit-sources-modal
```

The synchronization command remeasures both Modal volumes before and after
each bounded batch, excludes converted and retained canonical names, rejects
sources below 100 MiB, writes through an atomic `.tmp` path, verifies exact
upstream size, records SHA-256 provenance, and reserves conversion capacity.
