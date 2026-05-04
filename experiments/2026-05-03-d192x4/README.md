# d192x4

Date: 2026-05-03
Commit: 550ff9a
Config: configs/d192x4.toml
Command: uv run train-modal --config configs/d192x4.toml --gpu l4
W&B: https://wandb.ai/maxence-frenette/chess-engine-4/runs/yoyc0drl

## Result

Steps: 10000
Samples: 10240000
Final loss: 4.8904
Device: cuda

## Data

Modal Volume: chess-engine-4-training-data
Mounted path: /data/training_data

Files:

- training-run1-test80-20240401-0017.tar
- training-run1-test80-20240401-0117.tar

## Notes

- First full W&B-logged Modal run.
- Trained against result WDL labels and moves-left target.
- No shuffling, checkpointing, or mixed precision yet.
