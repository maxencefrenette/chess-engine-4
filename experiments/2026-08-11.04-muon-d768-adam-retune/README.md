# D768 Muon and d256 AdamW retune

## Goal

Audit the suspicious d256 Muon result by giving AdamW a matched local learning-
rate sweep, then test batched Muon at d768 with enough LR coverage to select a
stable arm.

All training comparisons use the same `0.055x` horizon, seed 1, model, accepted
samples, batch, loss, schedule, and hardware within a width. `EG_flops` values
were recomputed against one current canonical fit. Learning-rate selection
excludes any run with a recorded loss spike.

## Commands

```sh
uv run python experiments/2026-08-11.04-muon-d768-adam-retune/launch.py
uv run python experiments/2026-08-11.04-muon-d768-adam-retune/launch.py --launch
uv run python experiments/2026-08-11.04-muon-d768-adam-retune/launch.py --retry-interrupted --launch
uv run python experiments/2026-08-11.04-muon-d768-adam-retune/launch.py --adam-lower --launch
uv run python experiments/2026-08-11.04-muon-d768-adam-retune/launch.py --muon-upper --launch
uv run python experiments/2026-08-11.04-muon-d768-adam-retune/launch.py --muon-one --launch
uv run python experiments/2026-08-11.04-muon-d768-adam-retune/launch.py --muon-midpoint --launch
```

## D256 AdamW retune

| Optimizer | LR multiplier | EMA loss | `EG_flops` | Spikes | W&B |
| --- | ---: | ---: | ---: | ---: | --- |
| AdamW | `0.5x` | 3.388803 | 1.035x | 0 | [h2a10avr](https://wandb.ai/maxence-frenette/chess-engine-4/runs/h2a10avr) |
| AdamW | `0.625x` | **3.383249** | **1.067x** | 0 | [6nhqhoe5](https://wandb.ai/maxence-frenette/chess-engine-4/runs/6nhqhoe5) |
| AdamW | `0.75x` | 3.385384 | 1.054x | 0 | [i46si0a1](https://wandb.ai/maxence-frenette/chess-engine-4/runs/i46si0a1) |
| AdamW | `1x` | 3.409629 | 0.926x | 0 | [ufjguuuy](https://wandb.ai/maxence-frenette/chess-engine-4/runs/ufjguuuy) |
| AdamW | `1.25x` | 3.424759 | 0.855x | 0 | [6s6o3f06](https://wandb.ai/maxence-frenette/chess-engine-4/runs/6s6o3f06) |
| AdamW | `1.5x` | 3.426350 | 0.848x | 0 | [h828vgrr](https://wandb.ai/maxence-frenette/chess-engine-4/runs/h828vgrr) |
| Batched Muon | `1x` | **3.337754** | **1.377x** | 0 | [nfo80tar](https://wandb.ai/maxence-frenette/uncategorized/runs/nfo80tar) |

The suspicion was correct: the previous AdamW control was undertuned at this
short horizon. `0.625x`, or LR `0.000796875`, is the selected tested AdamW LR.
It raises AdamW `EG_flops` from 0.926x to 1.067x and reduces Muon's relative
training-FLOP gain from 48.7% to **29.1%**.

W&B wall times were too noisy for a trustworthy small realized-efficiency
claim: the selected Adam arm took an anomalous 60.9 seconds, while neighboring
Adam arms took 22-24 seconds. The matched 100-step synthetic full-training
benchmark is stable and isolates the optimizer comparison:

| Width | AdamW step | Muon step | Muon/Adam runtime | Muon/Adam `EG_flops` | Realized compute efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| d256 | 2.795 ms | 3.368 ms | 1.205x | 1.291x | **+7.1%** |

Thus the original 9.8% realized-dollar claim was overstated. A small d256
compute-efficiency gain survives under controlled timing, but it is only 7.1%
at this short single-seed horizon and is not robustly visible in production
wall time.

## D768 Muon

| Optimizer | LR multiplier | EMA loss | `EG_flops` | Spikes | W&B |
| --- | ---: | ---: | ---: | ---: | --- |
| AdamW | `1x` | **3.033531** | **1.120x** | 0 | [gk7to9xk](https://wandb.ai/maxence-frenette/chess-engine-4/runs/gk7to9xk) |
| Batched Muon | `0.25x` | 3.230159 | 0.312x | 0 | [4rwcl267](https://wandb.ai/maxence-frenette/chess-engine-4/runs/4rwcl267) |
| Batched Muon | `0.5x` | 3.100163 | 0.701x | 0 | [58gx4z2b](https://wandb.ai/maxence-frenette/chess-engine-4/runs/58gx4z2b) |
| Batched Muon | `0.75x` | 3.040616 | 1.064x | 0 | [wzbj17wh](https://wandb.ai/maxence-frenette/chess-engine-4/runs/wzbj17wh) |
| Batched Muon | `0.875x` | 3.022535 | 1.216x | 1 | [jdkqzxdg](https://wandb.ai/maxence-frenette/chess-engine-4/runs/jdkqzxdg) |
| Batched Muon | `1x` | 3.010074 | 1.336x | 2 | [ytz7xg3n](https://wandb.ai/maxence-frenette/chess-engine-4/runs/ytz7xg3n) |

The selected stable Muon LR is `0.75x`, or `0.0002475`. The higher arms improve
loss but are ineligible because LR tuning requires a spike-free run. Selected
Muon has 5.0% lower `EG_flops` than AdamW.

Matched batch-12,288 synthetic steps measured 7.726 ms for AdamW and 9.305 ms
for Muon. Combining convergence and runtime makes selected Muon **21.1% worse
in realized compute efficiency at d768**. The faster observed W&B wall time of
the 0.75x Muon arm was pipeline/container variance and is contradicted by the
controlled benchmark.

The d768 batched optimizer itself remains numerically sound: its optimizer-only
step matches PyTorch Muon within 0.000488 BF16 and takes 1.834 ms versus 3.057
ms for the reference implementation. The negative result is therefore not the
old serial-launch artifact.

## Interrupted attempts and cost

The initial AdamW `0.75x` and `1.25x` arms and Muon d768 `0.25x` arm were
terminated at steps 2,900, 4,860, and 6,990. W&B marked them finished during
cleanup, but they were excluded after checking `_step`; fresh retry runs supply
the reported results. Including these partial attempts, valid training arms
cost about $1.71. Benchmark cost was additional but small.

## Verdict

**The audit materially weakens the d256 Muon claim and rejects Muon at d768.**
Retuned AdamW closes much of the d256 convergence gap. Muon retains a small
7.1% controlled compute-efficiency gain at d256, but this is not robust enough
for promotion. At d768, the best stable Muon arm loses both training-FLOP and
controlled realized efficiency to AdamW.

Do not promote Muon or change the canonical AdamW LR from this short-horizon
audit. The d256 `0.625x` LR is specific evidence for `0.055x`; production LR
promotion requires the repository's intended horizon. Any further Muon work
should focus on explaining the width-dependent stability boundary, not another
general optimizer cutover attempt.
