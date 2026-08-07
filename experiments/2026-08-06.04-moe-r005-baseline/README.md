# MoE 0.05x Baseline

## Goal

Adopt a single `0.05x` Chinchilla training ratio for the `moe64a2` family and
refresh the canonical d128 through d1024 ladder. All runs used commit `38924d8`
plus the default-ratio change recorded by this experiment, the family
learning-rate law, and the configured RTX PRO 6000 or B200 backend.

## Results

| Width | Steps | LR | FLOPs | Loss | Policy top-1 | EG_flops | Spikes | Dead | W&B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| d128 | 4,144 | 2.8e-3 | 1.16e15 | 3.3077 | 38.04% | 1.49x | 0 | 0 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/fnbu2zs3) |
| d256 | 8,103 | 1.0e-3 | 1.41e16 | 3.0387 | 45.79% | 2.17x | 0 | 0 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/ujxunnwn) |
| d512 | 16,023 | 3.7e-4 | 1.91e17 | 2.8588 | 51.03% | 2.93x | 1 | 0 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/qv0at2vr) |
| d1024 | 29,888 | 1.3e-4 | 2.61e18 | 2.7507 | 54.17% | 2.85x | 3 | 0 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/u8fyelmb) |

`EG_flops` compares each candidate with the pre-refresh `0.02x` MoE frontier.
All four widths improve it materially and are promoted as the new canonical
ladder. Every run retained its final checkpoint under `/artifacts/checkpoints/`.

The d512 run had one isolated loss spike. The d1024 run had three early spikes,
then remained stable with zero dead experts; it is accepted by explicit decision
without a repair run. The d1024 loader exhausted the current training corpus at
29,888 of 31,863 configured steps, so its realized ratio is approximately
`0.0469x` rather than exactly `0.05x`.

## Commands

```sh
uv run train-modal --config configs/moe64a2.py --d-model 128 --wandb-name moe64a2-r005-refresh-d128
uv run train-modal --config configs/moe64a2.py --d-model 256 --wandb-name moe64a2-r005-refresh-d256
uv run train-modal --config configs/moe64a2.py --d-model 512 --wandb-name moe64a2-r005-refresh-d512
uv run train-modal --config configs/moe64a2.py --d-model 1024 --wandb-name moe64a2-r005-refresh-d1024
```

## Conclusion

Use `0.05x` Chinchilla as the single canonical training ratio for `moe64a2`.
The longer horizon lowers loss and improves physical-FLOPs efficiency throughout
the measured ladder while keeping every expert active. More training data is
required before a future d1024 refresh can complete the exact configured horizon.
