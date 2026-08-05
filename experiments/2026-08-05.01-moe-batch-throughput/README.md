# MoE Batch Throughput

## Goal

Measure whether larger batches improve B200 training throughput for the
`moe64a2` family after the static-dispatch optimization. The canonical recipe
uses `B = 32d`; this benchmark tests `B = 64d` and `B = 128d` from `d128`
through `d2048` without changing model or optimizer hyperparameters.

Each profile used 50 warmup steps and 500 measured steps. Jobs ran on Modal
B200s with eight CPU cores and eight parquet-loader threads. Representative
times are the mean of consistent repeats; noisy cases use the isolated rerun.

## Results

| Width | `32d` samples/s | `64d` samples/s | Gain | `128d` samples/s | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| d128 | 400k | 796k | 1.99x | 1.25M | 3.12x |
| d256 | 696k | 1.11M | 1.59x | 1.71M | 2.45x |
| d512 | 761k | 1.19M | 1.57x | 1.51M | 1.98x |
| d1024 | 624k | 792k | 1.27x | 981k | 1.57x |
| d2048 | 328k | 399k | 1.22x | OOM | - |

The larger batches improve throughput at every width. The gain declines as the
model widens because larger expert matrices already use the GPU more effectively.
`128d` is the throughput winner for `d128` through `d1024`, while `d2048` cannot
fit at `128d`.

`d2048/64d` is at the memory boundary: two profiles completed at about
329 ms/step, while one concurrent profile failed during CUDA graph capture due
to fragmentation with less than 2 GiB free. It is therefore not a robust
training configuration without reducing graph memory or batch size.

These are infrastructure results only. Promoting a larger batch to the model
recipe still requires loss experiments with learning-rate tuning because a fixed
sample budget would contain fewer optimizer steps.
