# Quantile Load Balancing

## Goal

Replace the canonical top-2 router auxiliary loss with quantile balancing (QB)
and compare statistical efficiency at the retained `moe64a2` d128, d256, and
d512 `0.05x` baselines. Model shape, seed, batch, accepted samples, task-loss
weights, learning rate, and training FLOPs are matched. The balancing mechanism
is the experimental variable: the baseline uses `router_aux = 0.01`, while QB
uses a detached per-expert bias and `router_aux = 0`.

The implementation follows `qb_dual_update` in NVIDIA Megatron-LM commit
`78901d8a71b92ed19e3e31e00815e6bde558e9de`. Transformer Engine was inspected at
commit `f07a86029d3831d5512e26c3e0791566a72e14f0`; the project remains pinned to TE
2.17.0, so QB computes its top-k indices and unbiased softmax probabilities in
PyTorch before reusing the existing TE/custom dispatch and expert kernels.
Runs used repository base commit `c65c7cc3db8f5d8aab52edeec363e276fffad6f6`
plus the accompanying quantile-balancing working-tree diff.

## Commands

```sh
uv run train-modal --config configs/moe64a2.py --d-model WIDTH \
  --training-ratio 0.05 --router-load-balancing quantile \
  --wandb-name quantile-moe64a2-dWIDTH-r0p05

uv run compare-run WANDB_URL \
  --best-runs experiments/best-runs-moe64a2.toml

# Fresh auxiliary-loss controls used the same command without QB:
uv run train-modal --config configs/moe64a2.py --d-model WIDTH \
  --training-ratio 0.05 --router-load-balancing aux_loss \
  --wandb-name quantile-control-rerun-moe64a2-dWIDTH-r0p05
```

## Statistical efficiency

| Width | Retained baseline | Fresh baseline | QB | Retained / fresh / QB loss | Retained / fresh / QB EG_flops | QB delta vs retained / fresh |
| --- | --- | --- | --- | ---: | ---: | ---: |
| d128 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/fnbu2zs3) | [run](https://wandb.ai/maxence-frenette/uncategorized/runs/vf746dtp) | [run](https://wandb.ai/maxence-frenette/uncategorized/runs/zios5rgb) | 3.307732 / 3.308292 / 3.312568 | 1.001x / 0.997x / 0.966x | -0.035x / -0.031x |
| d256 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/ujxunnwn) | [run](https://wandb.ai/maxence-frenette/uncategorized/runs/rf0ofkv7) | [run](https://wandb.ai/maxence-frenette/uncategorized/runs/88paxyjp) | 3.038671 / 3.043351 / 3.032097 | 0.973x / 0.922x / 1.050x | +0.077x / +0.128x |
| d512 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/qv0at2vr) | [run](https://wandb.ai/maxence-frenette/uncategorized/runs/x5wzq875) | [run](https://wandb.ai/maxence-frenette/uncategorized/runs/yc1aecop) | 2.858844 / 2.858559 / 2.847487 | 0.983x / 0.988x / 1.219x | +0.237x / +0.231x |

Loss is `loss/task[ema=0.99]`. All runs completed with zero dead experts. The
fresh baseline reproductions changed EG_flops by -0.004x, -0.051x, and +0.005x
at d128, d256, and d512 respectively. They therefore preserve the result: QB
loses at d128, wins at d256, and wins strongly at d512. Against the fresh
controls, QB policy top-1 EMA is 0.3841 vs 0.3776, 0.4604 vs 0.4539, and 0.5131
vs 0.5087.

The fresh controls recorded 0, 1, and 0 loss spikes, while QB recorded 0, 2,
and 2. The d256 and d512 QB spike intervals were isolated and returned
immediately to the prior trend, but the replicated controls make the stability
regression clearer and continue to block automatic promotion.

## Routing distributions

Final QB token counts are reported for every routed layer in the W&B summary.
The table condenses each 64-expert vector as `min-max (CV)`:

| Width | Layer 0 | Layer 1 | Layer 2 | Layer 3 |
| --- | --- | --- | --- | --- |
| d128 | 312-1054 (0.260) | 190-889 (0.232) | 324-840 (0.198) | 271-1129 (0.259) |
| d256 | 672-1361 (0.150) | 527-1447 (0.154) | 478-1614 (0.174) | 626-1378 (0.139) |
| d512 | 1344-2480 (0.099) | 957-2887 (0.152) | 1001-2874 (0.154) | 1074-2850 (0.125) |

QB prevents dead experts but does not produce exact per-batch balance in these
runs because the bias solved from one batch is applied to the next batch.

## Throughput

Fresh matched profiles used 20 warmup and 100 measured end-to-end steps. QB's
additional quantile work is not included in training-FLOP accounting, but it is
included in these wall times.

| Width | Backend | Baseline ms/step | QB ms/step | QB change |
| --- | --- | ---: | ---: | ---: |
| d128 | custom BF16, RTX PRO 6000 | 10.65 | 10.24 | -3.8% |
| d256 | custom BF16, A100 | 41.63 | 40.54 | -2.6% |
| d512 | TE MXFP8, B200 | 41.50 | 42.33 | +2.0% |

Peak allocated and reserved memory were effectively unchanged. The unfused QB
implementation is therefore adequate for evaluating statistical efficiency;
kernel optimization is not yet required to make the training path usable.

## Verdict

The fresh controls confirm that QB improves both EG_flops and policy top-1 at
d256 and d512, with a particularly strong d512 result. QB loses EG_flops at
d128, but improves policy top-1 there and removes a tuned loss weight.

After review, the project chose a full QB cutover based on the measured evidence
and the prior for the principled aux-loss-free method. The training strategy
switch and router auxiliary-loss weight were removed; all new MoE runs use QB.
The d256 and d512 QB runs are the only entries retained in
`best-runs-moe64a2.toml`. All auxiliary-loss runs, including the historical d128
statistical incumbent and the non-default-ratio allocation observations, were
deleted from the canonical best-runs file. Historical evidence remains in its
original experiment reports. This is an explicit decision override of the
spike-based automatic-promotion gate: the QB spikes and their immediate recovery
remain recorded above.

With only two current QB widths, the website reports the observed MoE points
without fitting display curves, and the budget planner excludes MoE until new QB
allocation observations provide enough evidence to refit its training-ratio law.

Safetensors export now includes each MoE layer's learned `router_qb_beta`, and
the lc0 ce4 CUDA router selects experts using `logit - beta` while computing
combine probabilities from the unbiased logits. Historical MoE files without
the tensor remain loadable with a zero beta. The production SM120 inference
library and lc0 fork compiled successfully, and the exported d256 winner passed
an end-to-end backend evaluation smoke on RTX PRO 6000.

The commands above are the historical experiment commands. After the cutover,
the canonical command has no routing option:

```sh
uv run train-modal --config configs/moe64a2.py --d-model WIDTH \
  --training-ratio 0.05
```
