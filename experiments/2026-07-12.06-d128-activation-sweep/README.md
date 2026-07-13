# d128 Activation Sweep

## Goal

Compare the dense model's activation functions at fixed `d128x4` shape, 4x
expansion, batch size 8,192, 10,024 steps, learning rate 1.1e-3, training data,
optimizer, and MXFP8 precision. Non-gated activations intentionally use fewer
parameters and physical FLOPs than the gated variants.

## Results

| Activation | Params | Training FLOPs | Loss | Loss + 1 SD | Policy top-1 | FLOPs efficiency | Modified-compute efficiency | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| GEGLU | 1,956,512 | 1.000e15 | **3.3611** | **3.4230** | **35.83%** | **1.030x** | **1.073x** | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/edy5sn69) |
| SwiGLU | 1,956,512 | 9.976e14 | 3.3624 | 3.4283 | 35.73% | 1.020x | 1.026x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/ypr6f3vn) |
| GELU | 1,694,368 | 8.702e14 | 3.3871 | 3.4539 | 34.78% | 0.925x | 0.940x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/cnna10mp) |
| SiLU | 1,694,368 | 8.680e14 | 3.4321 | 3.4939 | 33.91% | 0.612x | 0.671x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/4o2m9qae) |
| ReLU squared (`srelu`) | 1,694,368 | 8.685e14 | 3.4677 | 3.5285 | 32.85% | 0.444x | 0.504x | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/6q7jbyus) |

The physical-FLOPs efficiency multiplier is the primary comparison. The
existing `d128` incumbent remains stronger at 1.049x, so none of these runs
replaces the canonical recipe or best-run entry.

## Commands

```bash
uv run train-modal --config configs/dense/d128.toml --activation swiglu --wandb-name activation-d128-swiglu
uv run train-modal --config configs/dense/d128.toml --activation geglu --wandb-name activation-d128-geglu
uv run train-modal --config configs/dense/d128.toml --activation gelu --wandb-name activation-d128-gelu
uv run train-modal --config configs/dense/d128.toml --activation silu --wandb-name activation-d128-silu
uv run train-modal --config configs/dense/d128.toml --activation srelu --wandb-name activation-d128-srelu
```

All five jobs ran concurrently on B200s.

## Conclusion

GEGLU narrowly beat the fresh SwiGLU control on mean loss, uncertainty-adjusted
loss, policy accuracy, and both compute-efficiency fits. The margin is small,
and the run still trails the existing same-width incumbent, so SwiGLU remains
the default. GELU did not recover enough quality to justify its 13% lower FLOP
count. SiLU and squared ReLU were clearly worse.

If activation choice is revisited at larger scale, GEGLU is the only candidate
from this sweep worth carrying forward.
