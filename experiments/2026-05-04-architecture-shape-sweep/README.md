# Architecture Shape Sweep

Date: 2026-05-04
Commit: c71695a
GPU: Modal L4
Data: two t80 tar files in `chess-engine-4-training-data`

## Goal

Tune model shape for fixed FLOPs budgets. The budget configs are intended to
hold the best-so-far architecture and training hyperparameters for each target,
so this sweep only changed `d_model` and `depth` while keeping optimizer, batch
size, loss weights, and `mlp_ratio` fixed.

## Results

| Budget | Candidate | Params | Samples | Steps | Final loss | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1e13 | d64x2 | 678,342 | 2,321,408 | 2,267 | 6.1636 | https://wandb.ai/maxence-frenette/chess-engine-4/runs/1ts83gtm |
| 1e13 | d128x2 | 1,551,430 | 1,021,952 | 998 | 5.7040 | https://wandb.ai/maxence-frenette/chess-engine-4/runs/l2sdil6k |
| 1e14 | d160x3 | 2,369,062 | 6,749,184 | 6,591 | 4.6256 | https://wandb.ai/maxence-frenette/chess-engine-4/runs/orh9fvpn |
| 1e14 | d256x4 | 5,460,806 | 2,964,480 | 2,895 | 4.8572 | https://wandb.ai/maxence-frenette/chess-engine-4/runs/ibfl7i0j |
| 1e15 | d384x6 | 14,089,286 | 11,627,520 | 11,355 | 4.5764 | https://wandb.ai/maxence-frenette/chess-engine-4/runs/h1097nbe |
| 1e15 | d512x6 | 23,503,686 | 6,993,920 | 6,830 | 5.0542 | https://wandb.ai/maxence-frenette/chess-engine-4/runs/om8yx7uy |

## Config Updates

Updated the best-so-far budget configs:

- `configs/1e13.toml`: d128x2
- `configs/1e14.toml`: d160x3
- `configs/1e15.toml`: d384x6

## Notes

- Smaller models were better at 1e13 and 1e14 in this first pass.
- The 1e15 budget requires a larger model on the current two-file data slice;
  smaller models would run out of records before spending the target FLOPs.
- This is not a full architecture search. The next sweep should tune learning
  rate and batch size around these selected shapes.
