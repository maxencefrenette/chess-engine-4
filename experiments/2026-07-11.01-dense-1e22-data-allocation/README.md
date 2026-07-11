# Dense 1e22 Data Allocation

## Goal

Reallocate the `1e22` dense compute budget toward data. The previous baseline
used only 5.84 samples per parameter, while the smaller-budget frontier and the
working hypothesis both suggested a ratio near 20.

## Extrapolation

A power-law fit over the `1e18` through `1e21` dense winners predicted:

- 66.7M parameters at `1e22`
- 1.16B samples from the empirical sample-scaling fit
- approximately 1.33B samples when enforcing 20 samples per parameter

The selected `d896x6` model has 65,983,392 parameters. At batch 65,536, the
fixed compute budget produces 1,281,884,160 samples, or **19.43 samples per
parameter**.

## Training Data

The Modal volume was expanded from 96 to 168 tar files by adding all files from
`2024-04-12`, `2024-04-13`, and `2024-04-14`. After an initial parallel attempt,
the downloads were completed with one sequential worker to avoid placing
unnecessary load on the community-funded LCZero server.

The resulting dataset contains approximately 1.38B positions, enough to train
the selected recipe without exhausting the iterator or repeating data.

## Selected Result

| Recipe | Params | Samples | D/N | Loss | Loss + 1 SD | Policy top-1 | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Previous `d1152x6`, batch 24,576, LR 1.4e-4 | 106,068,896 | 619,585,536 | 5.84 | 2.9228 | 2.9494 | 48.70% | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/idabxu2m) |
| New `d896x6`, batch 65,536, LR 5e-4 | 65,983,392 | 1,281,884,160 | 19.43 | **2.8343** | **2.8597** | **51.08%** | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/eg1zhx9l) |

The new allocation improves loss upper 1 SD by **0.0897** and policy top-1 by
**2.38 percentage points**.

## Sweep Findings

- Shapes around 66M parameters were the correct regime. `d832x7` and `d896x6`
  were close at matched batch and LR, with `d896x6` winning after LR tuning.
- LR `1e-4` severely undertrained the smaller models. The score improved
  consistently through 2.2e-4, 3e-4, 4.2e-4, and 5e-4.
- LR 5.5e-4 had similar mean loss but substantially higher variance, producing
  a worse upper-1-SD score. This establishes a useful turnover near 5e-4.
- The result supports the approximately 20 samples-per-parameter hypothesis for
  the dense family at this scale.

## Command

```bash
uv run train-modal --config configs/dense/1e22.toml
```
