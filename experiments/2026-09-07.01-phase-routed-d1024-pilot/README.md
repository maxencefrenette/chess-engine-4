# Phase-routed dense crossover pilot

## Goal and method

Search for the low-compute crossover in the proposed "poor man's MoE": train
two fresh dense specialists on disjoint chess phases and compare them with one
fresh general model of the same width at the same total accepted samples and training
FLOPs. Positions with at least 20 pieces route to the high-piece specialist;
positions with at most 19 pieces route to the low-piece specialist. A
122,880-position, deep-offset, shard-stratified sample measured a
50.37%/49.63% high/low mix.

The experiment tested two allocations:

| Point | Width | Each specialist | General control | Positions per parameter, specialist / general |
| --- | ---: | ---: | ---: | ---: |
| `D = 0.05x` | 1024 | 275,038,208 positions | 550,076,416 positions | 2.5 / 5.0 |
| `D = 0.1x` | 1024 | 550,076,416 positions | 1,100,152,832 positions | 5.0 / 10.0 |
| `D = 0.2x` | 768 | 636,370,944 positions | 1,272,741,888 positions | 10.0 / 20.0 |
| `D = 0.5x` | 512 | 746,061,824 positions | 1,492,123,648 positions | 25.0 / 50.0 |

At each point, the two specialists together processed exactly the same number
of accepted positions and training FLOPs as the general control.

## Commands

The runs used base commit `c1405c699e5417ee8fe55fa868f8122e3eedf4fc`
plus the experiment-only loader and logging changes in the working tree.

All runs used `uv run train-modal --sampling-rate 1 --piece-count-cutoff 20`,
with the following arguments:

| Point | Width | Role | Config | Ratio | Mode | Additional arguments |
| --- | ---: | --- | --- | ---: | --- | --- |
| `D = 0.05x` | 1024 | Low | `phase_routed_d1024_pilot.py` | 0.05 | `low` | `--wandb-name phase-routed-d1024-low-r0p05` |
| `D = 0.05x` | 1024 | High | `phase_routed_d1024_pilot.py` | 0.05 | `high` | `--wandb-name phase-routed-d1024-high-r0p05` |
| `D = 0.05x` | 1024 | General | `phase_routed_d1024_pilot.py` | 0.1 | `all` | `--log-piece-count-losses --detach --wandb-name phase-routed-d1024-general-r0p1-restart2` |
| `D = 0.1x` | 1024 | Low | `phase_routed_d1024_pilot.py` | 0.1 | `low` | `--detach --wandb-name phase-routed-d1024-low-r0p1` |
| `D = 0.1x` | 1024 | High | `phase_routed_d1024_pilot.py` | 0.1 | `high` | `--detach --wandb-name phase-routed-d1024-high-r0p1` |
| `D = 0.1x` | 1024 | General | `dense.py` | 0.2 | `all` | `--log-piece-count-losses --detach --wandb-name phase-routed-d1024-general-r0p2` |
| `D = 0.2x` | 768 | Low | `dense.py` | 0.2 | `low` | `--detach --wandb-name phase-routed-d768-low-r0p2` |
| `D = 0.2x` | 768 | High | `dense.py` | 0.2 | `high` | `--detach --wandb-name phase-routed-d768-high-r0p2` |
| `D = 0.2x` | 768 | General | `dense.py` | 0.4 | `all` | `--log-piece-count-losses --detach --wandb-name phase-routed-d768-general-r0p4` |
| `D = 0.5x` | 512 | Low | `dense.py` | 0.5 | `low` | `--detach --wandb-name phase-routed-d512-low-r0p5` |
| `D = 0.5x` | 512 | High | `dense.py` | 0.5 | `high` | `--detach --wandb-name phase-routed-d512-high-r0p5` |
| `D = 0.5x` | 512 | General | `dense.py` | 1.0 | `all` | `--log-piece-count-losses --detach --wandb-name phase-routed-d512-general-r1` |

Config paths are under `configs/`; each row adds `--config`,
`--d-model`, `--training-ratio`, and `--piece-count-mode` from the table. The canonical
ratio-specific batch and learning-rate recipes were left unchanged.

## Results

`EG_flops`: not applicable across the specialists' different data
distributions. Promotion verdict: no promotion.

### Individual runs

| Point | Run | Accepted positions | Training FLOPs | EMA loss | Spikes | W&B |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `D = 0.05x` | Low specialist (`<=19`) | 275,038,208 | 1.8248e17 | 2.93350 | 0 | [svdr9j7p](https://wandb.ai/maxence-frenette/uncategorized/runs/svdr9j7p) |
| `D = 0.05x` | High specialist (`>=20`) | 275,038,208 | 1.8248e17 | 3.11277 | 0 | [wpcv1r4q](https://wandb.ai/maxence-frenette/uncategorized/runs/wpcv1r4q) |
| `D = 0.05x` | General control | 550,076,416 | 3.6497e17 | 2.98952 | 0 | [h6s12yct](https://wandb.ai/maxence-frenette/uncategorized/runs/h6s12yct) |
| `D = 0.1x` | Low specialist (`<=19`) | 550,076,416 | 3.6497e17 | 2.82495 | 0 | [nhha8cse](https://wandb.ai/maxence-frenette/uncategorized/runs/nhha8cse) |
| `D = 0.1x` | High specialist (`>=20`) | 550,076,416 | 3.6497e17 | 3.03249 | 0 | [pp5v4jyv](https://wandb.ai/maxence-frenette/uncategorized/runs/pp5v4jyv) |
| `D = 0.1x` | General control | 1,100,152,832 | 7.2994e17 | 2.76821 | 0 | [ff43cub0](https://wandb.ai/maxence-frenette/uncategorized/runs/ff43cub0) |
| `D = 0.2x` | d768 low specialist (`<=19`) | 636,370,944 | 2.4460e17 | 2.67104 | 0 | [aojvjunv](https://wandb.ai/maxence-frenette/uncategorized/runs/aojvjunv) |
| `D = 0.2x` | d768 high specialist (`>=20`) | 636,370,944 | 2.4460e17 | 2.90192 | 0 | [nymfkshy](https://wandb.ai/maxence-frenette/uncategorized/runs/nymfkshy) |
| `D = 0.2x` | d768 general control | 1,272,741,888 | 4.8921e17 | 2.78136 | 0 | [vl8i70xp](https://wandb.ai/maxence-frenette/uncategorized/runs/vl8i70xp) |
| `D = 0.5x` | d512 low specialist (`<=19`) | 746,061,824 | 1.3485e17 | 2.68782 | 0 | [ihh9v7ev](https://wandb.ai/maxence-frenette/uncategorized/runs/ihh9v7ev) |
| `D = 0.5x` | d512 high specialist (`>=20`) | 746,061,824 | 1.3485e17 | 2.92077 | 0 | [2ae0fq44](https://wandb.ai/maxence-frenette/uncategorized/runs/2ae0fq44) |
| `D = 0.5x` | d512 general control | 1,492,123,648 | 2.6970e17 | 2.81214 | 0 | [rplgx9nj](https://wandb.ai/maxence-frenette/uncategorized/runs/rplgx9nj) |

### Direct regional comparison

| Point | Region | Specialist EMA | General regional EMA | Specialist minus general |
| --- | --- | ---: | ---: | ---: |
| `D = 0.05x` | High (`>=20`) | 3.11277 | 3.10044 | +0.01233 |
| `D = 0.05x` | Low (`<=19`) | 2.93350 | 2.90547 | +0.02803 |
| `D = 0.1x` | High (`>=20`) | 3.03249 | 2.90343 | +0.12906 |
| `D = 0.1x` | Low (`<=19`) | 2.82495 | 2.66306 | +0.16189 |
| `D = 0.2x` | High (`>=20`) | 2.90192 | 2.90350 | -0.00158 |
| `D = 0.2x` | Low (`<=19`) | 2.67104 | 2.67064 | +0.00040 |
| `D = 0.5x` | High (`>=20`) | 2.92077 | 2.93579 | -0.01502 |
| `D = 0.5x` | Low (`<=19`) | 2.68782 | 2.69859 | -0.01077 |

Weighting each system by the sampled 50.37%/49.63% high/low mixture gives:

| Point | Routed loss | Regional-control loss | Routed deficit | Relative deficit |
| --- | ---: | ---: | ---: | ---: |
| `D = 0.05x` | 3.02380 | 3.00368 | +0.02012 | 0.67% |
| `D = 0.1x` | 2.92949 | 2.78413 | +0.14535 | 5.22% |
| `D = 0.2x` | 2.78733 | 2.78793 | -0.00060 | -0.02% |
| `D = 0.5x` | 2.80516 | 2.81807 | -0.01291 | -0.46% |

## Verdict

The d1024 routed pair lost at `D = 0.05x` and `D = 0.1x`. At `D = 0.2x`, the
d768 routed pair was marginally better by 0.00060 loss (0.02%): the high
specialist improved by 0.00158 while the low specialist regressed by 0.00040.
This is the first non-negative point, but the margin is too small for one seed
to establish a crossover. It also changes width, so it does not directly
locate the d1024 crossover.

At `D = 0.5x`, the d512 routed pair improved by 0.01291 loss (0.46%), with both
specialists beating the general model on their own regions. This is a stronger
positive result, although changing width means it still does not locate the
d1024 crossover. Replication should precede inference integration or Elo.
