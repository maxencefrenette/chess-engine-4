# Position subsampling for value diversity

## Status

Blocked at the required source-capacity and storage gate before dataset materialization or
training. No training run, W&B run, checkpoint export, tournament, canonical data change, or
sampling-rule promotion was performed.

Task: `01kzg0rdwf`  
Branch: `codex/position-subsampling-01kzg0rdwf`  
Base commit: `316c327da2612f5b71e9776a9ae8d3fc773c216d`

## Required design

The requested matched comparison fixes the model, optimizer and loss recipe, seed, batch size,
steps, samples, and training FLOPs while varying deterministic within-game position retention at
`1.0`, `0.5`, and `0.25`. Lower-retention arms must cover more source games and may not repeat rows.
All data and artifacts must remain isolated from canonical `/parquet` and from the corpus-expansion
task.

The prospective deterministic rule used for capacity accounting retains
`ceil(game_positions * numerator / denominator)` positions from every nonempty game. A production
materializer would choose one position from each of that many equal-width ply strata using a
stable keyed hash of source archive, game member, experiment seed, and stratum. This gives each
game the requested weight to within one row, retains short games, spreads samples over the whole
game, and avoids favoring long games beyond the requested rate. The final rule was not implemented
or promoted because the capacity gate failed first.

## Cost-limited model selection

The current planner was run against an intentionally nonbinding ten-billion-row availability so
the selection was constrained by the strict per-run dollar ceiling rather than by the current
corpus:

```sh
uv run plan-budget 1.499999 --assume-samples 10000000000 --bootstrap-samples 20
```

It selected the canonical `moe64a2` family at `d_model=512`, depth 8, MXFP8 Transformer Engine
recipe, batch size 65,536, seed 1, and the existing policy/value/moves-left plus router-auxiliary
loss recipe. The allocation is slightly beyond the observed `0.05x` anchor but
within the planner's configured 2x extrapolation limit; all 20 bootstrap fits selected it.

| Quantity | Planned value |
| --- | ---: |
| Training ratio | 0.0621746945093x |
| Steps | 19,925 |
| Samples per arm | 1,305,804,800 |
| Training FLOPs per sample | 181,555,318 |
| Training FLOPs per arm | 2.370758057099264e17 |
| Measured steady-state step time | 40.895007764 ms |
| Planned steady-state runtime | 814.833029698 s |
| B200 rate | $0.001736/s |
| Eight-CPU rate | $0.0001048/s |
| Planned GPU+CPU cost per arm | $1.499944641 |
| Planner loss prediction | 2.8398 (80% bootstrap interval 2.8322-2.8445) |

The independent arithmetic is
`19,925 * 0.040895007764 * (0.001736 + 8 * 0.0000131) = $1.499944641`, which is
strictly below `$1.50`. The three training runs would total `$4.499833923` in steady-state GPU+CPU
charges before startup, memory, export, or evaluation.

## Audited source capacity

The corpus task retained and manifested the eight clean source archives from
`training-run1-test80-20240428-1417.tar` through `-2217.tar`, excluding the anomalous 1.45 MB
`-1817.tar`. The immutable provenance records are under `/source-manifests` on the
`chess-engine-4-training-data` Modal Volume. The eight archives contain 535,627 validated game
members, 58,681,470 v6 positions, and 12,050,606,080 source bytes. No repeated canonical game IDs
were detected in that set.

The `0.25` arm alone requires at least 1,305,804,800 retained rows. On the initial eight sources,
the proposed per-game ceiling rule can retain no more than
`floor((58,681,470 + 3 * 535,627) / 4) = 15,072,087` rows, at most 1.15% of the requirement.
The exact acquisition criterion sent to the corpus task was therefore:

```text
sum(ceil(game_positions / 4)) >= 1,305,804,800
```

Nominally this needs 5,223,219,200 raw source positions. The exact audited density used by the
corpus task is `12,050,606,080 / 58,681,470 = 205.356241` compressed bytes per raw position, so the
required sources project to 1,072,620,659,447 bytes (998.956 GiB) before any output reserve.

## Exact blocker

The operational ceiling is 900 GiB, or 966,367,641,600 bytes, across the training-data and
artifacts Volumes. Pre-source combined usage was 690,294,698,209 bytes. The corpus task reserves
one full source-sized output allocation for every unconverted archive, so acquisition may use at
most `floor((966,367,641,600 - 690,294,698,209) / 2) = 138,036,471,695` source bytes
(128.556 GiB).

At the audited density, that policy supports about 672,180,554 raw positions or 168,045,139
nominal quarter-rate rows. This is 7.7706x short of the required 1,305,804,800 rows. The required
sources and conservative output reserve project to 1,997.912 GiB; fitting them under the policy
would require at most 26.427 compressed bytes per raw position, an unsupported 7.77x compression
improvement.

Independently, three matched datasets at the audited canonical density of 86.770565 bytes per row
would require approximately 339,916,260,827 bytes, without provenance columns, checkpoints, source
conversion reserve, or tournament outputs. The nominal source archives plus those three datasets
would require about 1.413 TB of additional storage. The source archives alone exceed the entire
900 GiB combined ceiling.

Consequently the requested strongest sub-$1.50 configuration cannot receive matched unique rows
for the `0.25` treatment under the declared storage policy. Training on fewer rows, repeating rows,
choosing a weaker sample-limited model after the fact, or overwriting canonical data would each
violate the experiment design. The experiment stopped at this gate.

Acquisition stopped after 17 complete source archives had been hashed and manifested. The final
audit at corpus-task commit `c678bcb` records 25,723,576,320 source bytes, 1,143,453 games,
125,250,708 v6 positions, and zero duplicate canonical game IDs. Live combined Modal Volume usage
was 716,018,279,187 bytes, leaving 250,349,362,413 bytes below the 900 GiB operational ceiling.
The corpus task removed only the incomplete successor `.tmp`; it preserved all completed sources
and manifests. Its retained evidence is
`experiments/2026-08-08.01-training-corpus-capacity/README.md` and `sources.toml` on branch
`codex/expand-final-training-corpus`. No canonical conversion or `training-data.toml` change
occurred.

## Training and tournament results

Not run. There are no W&B URLs, checkpoints, task-loss components, policy top-1 measurements,
stability results, realized costs, exported models, 800-node searched games, WDL counts, Elo
intervals, runtime, or inference-throughput measurements to report.

If the capacity constraint is later changed, the tournament must use all three valid checkpoints
in a connected comparison at approximately 800 nodes per move with mirrored openings. Raw opening
pairs must be retained for the paired estimator being developed under task `01kzg0rdd6`; the
existing unpaired estimator must not be independently rewritten here.

## Recommendation

Do not retain or reject any sampling rate on experimental grounds: no treatment was trained.
Keep canonical configs and data unchanged. Resume only after the user chooses one of these scope
changes:

1. authorize deletion of verified temporary raw archives after each isolated treatment is
   materialized, enabling a sequential streaming design whose exact peak-storage plan must be
   rechecked before any deletion or launch; or
2. explicitly authorize a smaller, storage-limited model and matched row target.

Supplying enough external source/data storage for the original design would also clear the
blocker. Until then, preserve all 17 completed sources and keep this task stopped.
