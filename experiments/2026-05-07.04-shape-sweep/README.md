# Shape Sweep

Date: 2026-05-07

## Goal

Lightly retune model shape for the `1e14`, `1e15`, and `1e16` step-adjusted compute budgets. The first wave tested 21 shapes across all budgets. A second wave added 10 follow-up runs around the best first-wave regions.

## Commands

The launched commands are listed in `commands.txt`.

## Winners

| Budget | Previous shape | Previous loss | Best shape | Params | Loss | Policy top-1 | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1e14 | d48x1 | 4.7959 | d32x1 | 303,206 | 4.6455 | 0.2270 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/qztrcwzl) |
| 1e15 | d64x5 | 4.5407 | d80x3 | 955,062 | 4.4384 | 0.2952 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/bpwyq4u1) |
| 1e16 | d192x4 | 4.2711 | d160x6 | 3,291,142 | 4.2263 | 0.3463 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/0j9qoqiy) |

## Ranked Results

### 1e14

| Shape | Wave | Params | Loss | Policy top-1 | Final loss | W&B |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| d32x1 | shape | 303,206 | 4.6455 | 0.2270 | 3.9809 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/qztrcwzl) |
| d32x2 | followup | 315,526 | 4.6545 | 0.2431 | 4.2458 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/wvdisod9) |
| d28x1 | followup | 264,194 | 4.7107 | 0.2276 | 5.0499 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/py3g6vqs) |
| d48x2 | shape | 490,790 | 4.7162 | 0.2378 | 4.3592 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/ru492dit) |
| d56x1 | shape | 545,342 | 4.7780 | 0.2322 | 4.7440 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/9zxzefbh) |
| d48x1 | shape | 463,094 | 4.8024 | 0.2350 | 4.6945 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/tjbz4djv) |
| d24x1 | followup | 225,566 | 4.8677 | 0.2303 | 5.2223 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/st36df3h) |
| d40x1 | shape | 382,382 | 4.9406 | 0.2290 | 5.3676 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/ju53m7ba) |

### 1e15

| Shape | Wave | Params | Loss | Policy top-1 | Final loss | W&B |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| d80x3 | shape | 955,062 | 4.4384 | 0.2952 | 4.7124 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/bpwyq4u1) |
| d96x3 | shape | 1,200,998 | 4.4674 | 0.3064 | 3.8535 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/d1b8zzh2) |
| d112x3 | shape | 1,465,366 | 4.4686 | 0.3019 | 4.7999 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/c7ugbjkx) |
| d80x2 | followup | 878,182 | 4.4789 | 0.2986 | 4.8241 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/bvczqnqf) |
| d96x2 | shape | 1,090,310 | 4.4906 | 0.2979 | 3.8535 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/5yizivr3) |
| d128x3 | shape | 1,748,166 | 4.4907 | 0.2965 | 3.7557 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/u00pqi2a) |
| d72x3 | followup | 839,006 | 4.4918 | 0.2994 | 4.8358 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/opt013ii) |
| d112x2 | shape | 1,314,726 | 4.4947 | 0.2917 | 3.8586 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/xwzgh19p) |
| d96x4 | shape | 1,311,686 | 4.4951 | 0.2962 | 4.3363 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/lqegy4cl) |
| d80x4 | followup | 1,031,942 | 4.4954 | 0.2997 | 4.6973 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/1h0tmxxh) |
| d128x2 | shape | 1,551,430 | 4.5012 | 0.3023 | 4.7023 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/y9lwd1fh) |
| d88x3 | followup | 1,075,726 | 4.5634 | 0.3022 | 3.9646 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/gjo7qdqd) |

### 1e16

| Shape | Wave | Params | Loss | Policy top-1 | Final loss | W&B |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| d160x6 | shape | 3,291,142 | 4.2263 | 0.3463 | 4.1504 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/0j9qoqiy) |
| d160x5 | shape | 2,983,782 | 4.2376 | 0.3521 | 4.4302 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/t9g2qa1e) |
| d144x8 | shape | 3,294,278 | 4.2419 | 0.3473 | 4.3228 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/ytb2svsf) |
| d160x4 | shape | 2,676,422 | 4.2421 | 0.3483 | 4.2679 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/kyhopniv) |
| d168x6 | followup | 3,552,374 | 4.2504 | 0.3534 | 4.6838 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/kdvc58rm) |
| d192x3 | shape | 3,063,686 | 4.2704 | 0.3531 | 4.4043 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/1c272u7o) |
| d128x10 | shape | 3,125,318 | 4.2731 | 0.3474 | 4.1437 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/d124n2q6) |
| d192x4 | shape | 3,506,246 | 4.2734 | 0.3534 | 4.1945 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/i3pzi11l) |
| d176x4 | shape | 3,079,046 | 4.2745 | 0.3534 | 4.7265 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/7i0j8583) |
| d160x7 | followup | 3,598,502 | 4.2880 | 0.3526 | 4.7116 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/zpy0zcxf) |
| d152x6 | followup | 3,039,126 | 4.2982 | 0.3500 | 4.2589 | [run](https://wandb.ai/maxence-frenette/chess-engine-4/runs/nrvpva2t) |

## Takeaways

- `1e14` prefers a much smaller model than the previous d48x1 baseline. The best run was d32x1; d32x2 was close, while d24x1 and d28x1 were too small.
- `1e15` improved by moving from the old d64x5 shape to d80x3. Larger models around d96-d128 were close but did not beat d80x3 by tail loss.
- `1e16` improved by moving from d192x4 to a deeper/narrower d160x6. The d160 family dominated this sweep, while going wider again did not help.
- Best-run configs and `experiments/best-runs-mlp.toml` now point to d32x1, d80x3, and d160x6.
