# Parameter-matched dense versus BT4 inference

Median LCZero `backendbench` results from three runs on an RTX PRO 6000:

| Batch | Model | Parameters | Positions/s | Latency | Dense speedup |
| ---: | --- | ---: | ---: | ---: | ---: |
| 256 | random dense d1344x8 | 185.68M | 345,960 | 0.740 ms | **31.0x** |
| 256 | BT4-1740 | ~190M | 11,146 | 22.970 ms | - |
| 2048 | random dense d1344x8 | 185.68M | 482,126 | 4.248 ms | **49.0x** |
| 2048 | BT4-1740 | ~190M | 9,849 | 207.900 ms | - |

The closest existing dense export was 11% smaller than BT4, so the benchmark
used a random d1344x8 network, 2.27% below BT4's published parameter count.
Results include LCZero input/output handling and transfers, but exclude search.
