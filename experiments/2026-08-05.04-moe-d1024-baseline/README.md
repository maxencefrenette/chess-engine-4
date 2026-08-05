# MoE d1024 Baseline

## Goal

Establish the first `d1024` baseline for the alternating-layer `moe64a2`
family. Before the full run, compare the fused dynamic token dispatcher with
the static dispatcher and CUDA graph path at the exact canonical shape.

The training-data volume contained 3,949,735,220 positions across 480 Parquet
shards. The run required 1,670,512,640 positions, or 42.3% of the available
data.

## Dispatcher Profile

Both profiles used `d1024`, batch size 131,072, 50 warmup steps, and 500
measured steps on a B200.

| Dispatcher | Wall time / step | Train GPU time / step | End-to-end MFU |
| --- | ---: | ---: | ---: |
| Fused dynamic | **138.09 ms** | **131.97 ms** | **14.03%** |
| Static + CUDA graph | 140.54 ms | 135.79 ms | 13.78% |

Fused dynamic dispatch is 1.77% faster end to end and remains the d1024 path.
The exposed GPU idle gap was only 0.27 ms/step, so data loading was not the
bottleneck.

## Training Result

| Width | Parameters | Batch | Steps | Training FLOPs | Loss | Policy top-1 | Runtime | Dead experts | W&B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| d1024 | 1.671B | 131,072 | 12,745 | 1.111e18 | 2.8257 | 52.22% | 1,795s | 0 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/wmpvk8ot) |

The run recorded one isolated loss spike around step 7,823. Loss recovered
immediately, routing retained zero dead experts, and the result is accepted as
the canonical d1024 baseline with the spike explicitly annotated in
`experiments/best-runs-moe64a2.toml`.

The final checkpoint is stored at:

```text
/artifacts/checkpoints/moe64a2-d1024-baseline-final.pt
```

At the B200 price used by this project, the recorded runtime costs approximately
`$3.12`.

## Command

```sh
uv run train-modal --config configs/moe64a2.py --d-model 1024 \
  --wandb-name moe64a2-d1024-baseline
```
