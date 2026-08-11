# NVFP4 inference crossover

## Goal

Locate whether Transformer Engine NVFP4 inference overtakes MXFP8 for a
potential large dense model at batch 4096. This is an inference-only benchmark;
it does not run backward, an optimizer, the training input pipeline, or a
training loop.

## Context

- Project base commit: `be84b3ccc30b897f0af867e1af13233ee6ae4523`.
- Benchmark harness: `bench_nvfp4_inference_crossover.py`, uncommitted in the
  benchmark worktree at measurement time.
- GPU: NVIDIA B200.
- Software: PyTorch `2.11.0+cu130`, Transformer Engine `2.17.0`.
- Transformer Engine source reference:
  `07e281f2ba93b61c5ab6145dbdaa2a768b888e19`.
- Model: dense, depth 8, SwiGLU, expansion ratio 4, widths d2048, d3072,
  and d4096.
- Input: seeded synthetic BF16 planes with shape `[4096, 112, 8, 8]`.
- Measurement: cached quantized weights, 10 paired warmups, 30 paired timed
  forwards, alternating precision order on the same B200.
- The timed forward includes the input projection, all residual MLP blocks,
  final norm, and all output heads. It excludes lc0 runtime and input packing.
- W&B URL: N/A (focused inference benchmark; no training run).
- `EG_flops`: N/A (no loss curve or training run).

## Command

```sh
uv run python bench_nvfp4_d2048.py
uv run python bench_nvfp4_inference_crossover.py
```

The d2048 baseline used the predecessor harness before it was generalized and
renamed. Two independent Modal runs were retained for each sweep:

- `ap-RJ419vxvjTQszCRis8tnG7`
- `ap-9bX5JDyNF0FhBR0WdMzOE0`
- `ap-P1lc4ZbTrdlek1UeYR1DpF`
- `ap-vliDnu6149WOSrW0hNTopB`

## Results

`NVFP4 speedup` is MXFP8 median latency divided by NVFP4 median latency, so a
value above one means NVFP4 is faster.

| Width | Modal run | MXFP8 median | NVFP4 median | MXFP8 positions/s | NVFP4 positions/s | NVFP4 speedup |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| d2048 | `ap-RJ419vxvjTQszCRis8tnG7` | 2.6619 ms | 3.6568 ms | 1,538,739 | 1,120,095 | 0.7279x |
| d2048 | `ap-9bX5JDyNF0FhBR0WdMzOE0` | 2.3293 ms | 3.0204 ms | 1,758,459 | 1,356,090 | 0.7712x |
| d3072 | `ap-P1lc4ZbTrdlek1UeYR1DpF` | 4.3310 ms | 3.5660 ms | 945,738 | 1,148,641 | 1.2145x |
| d3072 | `ap-vliDnu6149WOSrW0hNTopB` | 4.3305 ms | 3.6504 ms | 945,847 | 1,122,064 | 1.1863x |
| d4096 | `ap-P1lc4ZbTrdlek1UeYR1DpF` | 7.0220 ms | 5.2076 ms | 583,308 | 786,540 | 1.3484x |
| d4096 | `ap-vliDnu6149WOSrW0hNTopB` | 6.6860 ms | 5.0528 ms | 612,623 | 810,645 | 1.3232x |

NVFP4 changed from `0.728-0.771x` at d2048 to `1.186-1.215x` at d3072. The
measured crossover is bounded between d2048 and d3072; this experiment does not
locate the exact intermediate width.

At d3072, NVFP4 delivered 18.6-21.5% higher throughput. At d4096, its advantage
grew to 32.3-34.8%. The two independent paired runs agree on both the direction
and approximate size of the advantage.

Exact JSON for all six width measurements is retained in `results.json`.

## Verdict

Performance result: NVFP4 crosses over before or at d3072 for this batch-4096
Transformer Engine inference path and wins more decisively at d4096.

Promotion: no. These widths are potential shapes rather than canonical dense
recipes, numerical quality was not evaluated, and this benchmark does not
measure the standalone lc0 inference backend. Use MXFP8 at d2048 and prefer
NVFP4 for further TE inference investigation at d3072 and d4096. A focused
d2560 or d2816 probe would be needed to narrow the crossover boundary.
