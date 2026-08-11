# Dense and MoE scaling points

## Verdict

Dense d768/d1280 and MoE64A2 d768 are promoted as stable new-width points.
Dense d768 and MoE d768 beat their family trends; dense d1280 is below trend
but has no incumbent and completed without spikes. Per the revised scope, dense
d1536/d2048 and MoE d1280/d1536 are not in the ladders.

| Family | Width | Ratio | EMA loss | EG_flops | Spikes | Verdict | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| dense | d768 | 0.2x | 2.927340 | 1.286x | 1 | promote, beats trend | [5w1kgfp2](https://wandb.ai/maxence-frenette/uncategorized/runs/5w1kgfp2) |
| dense | d1280 | 0.2x | 2.858356 | 0.672x | 0 | promote, below trend | [doglducv](https://wandb.ai/maxence-frenette/uncategorized/runs/doglducv) |
| moe64a2 | d768 | 0.05x | 2.778502 | 1.291x | 3 | promote, beats trend | [uysyhdm4](https://wandb.ai/maxence-frenette/uncategorized/runs/uysyhdm4) |

The dense d768 spike occurred at step 22,920; its EMA improved from 2.983869
to 2.927340 afterward. MoE spikes were recorded at steps 2,460 and 17,710;
its EMA improved from 3.192129 and 2.853482 to 2.778502. It finished with zero
dead experts.

## Kernel and recipe selection

Source commit: `13f4eca587ef4cf2ad22f3f086ee5ecb37dd876a`.
Local references were TransformerEngine `f07a8602`, mixture-of-kittens
`6438bf48`, and DeepGEMM `559d79fb`. The d768/d1280 SM100 MXFP8
RMSNorm/SwiGLU paths passed forward/backward numerical thresholds alongside
d512/d1024 neighbors. Transformer Engine was retained: custom was slower
end-to-end (d768 14.41 vs 12.79 ms; d1280 49.40 vs 44.60 ms in focused
profiles). B200 MXFP8 also beat B200 BF16 and the viable BF16 GPU alternatives.
A100 and RTX-PRO-6000 MoE profiles failed CUDA graph capture; H200 was dominated
by H100 at equal BF16 compute without a memory requirement.

Canonical 50-warmup/500-step profiles include overlapped loader time, eight CPU
cores, measured peak memory, and the Modal price basis of `$0.001736/s` GPU plus
`$0.0001048/s` CPU:

| Family | Width | Step | Samples/s | Peak allocated/reserved | Training cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | d768 | 12.37 ms | 1.987M | 1.36/7.34 GiB | $0.59 at 0.2x |
| dense | d1280 | 45.02 ms | 1.092M | 3.30/21.58 GiB | $2.85 at 0.2x |
| moe64a2 | d768 | 79.65 ms | 1.234M | 15.55/41.75 GiB | $3.51 at 0.05x |

## Commands

```sh
uv run benchmark-training-modal --widths 512 768 1024 1280 1536 --gpu B200 --quantization-recipe mxfp8 --level layer --warmup 10 --iterations 30 --json
uv run train-modal --config configs/dense.py --d-model 768 --training-ratio 0.2 --wandb-name scaling-dense-d768-r0p2
uv run train-modal --config configs/dense.py --d-model 1280 --training-ratio 0.2 --wandb-name scaling-dense-d1280-r0p2
uv run train-modal --config configs/moe64a2.py --d-model 768 --training-ratio 0.05 --wandb-name scaling-moe64a2-d768-r0p05
uv run compare-run WANDB_URL
uv run compare-run WANDB_URL --best-runs experiments/best-runs-moe64a2.toml
```

The d1536 kernel check preceded its cancellation; it was not retained in the
recipe or trained. Full raw histories and launch evidence remain in the linked
W&B runs and Modal applications `ap-X7PSOMp1mrhyEExkniKOs3` (dense d1280) and
`ap-lfQ7WJQZC3uBTxaylhHGdF` (MoE d768), plus the d768 W&B artifact lineage.
