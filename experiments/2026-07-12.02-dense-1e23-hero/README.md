# Dense 1e23 Hero Run

## Goal

Train the first dense `1e23` baseline from the fitted model, data, batch-size,
and learning-rate trends. This is the first run at this scale rather than a
hyperparameter sweep.

## Recipe

| Item | Value |
| --- | ---: |
| Model | `d1472x8` |
| Parameters | 221,452,576 |
| Batch size | 131,072 |
| Learning rate | 3e-4 |
| Steps | 23,919 |
| Samples | 3,135,111,168 |
| Compute budget | `1e23` |
| Physical FLOPs | `4.181e18` |
| Precision | MXFP8 |
| GPU | B200 |

The batch-size extrapolation originally appeared as 24,576 because the
rounding ladder was capped at that value. Extending the ladder produced the
trend-consistent batch of 131,072 and reduced the planned step count from an
incorrect 141K to 23,919.

## Result

| Loss | Loss + 1 SD | Policy top-1 | Runtime | MFU | W&B |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2.7150 | 2.7501 | 54.95% | 59m 28s | 26.51% | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/zi6v0iq8) |

The fitted pre-run loss prediction was 2.7168, only 0.0018 above the observed
EMA loss. Training remained stable through the final step and completed in
less than one hour.

The committed final checkpoint is stored in the Modal artifacts volume at:

```text
/artifacts/checkpoints/dense-1e23-hero-final.pt
```
