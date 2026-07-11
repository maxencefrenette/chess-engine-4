# Dense MXFP8 vs BF16 Validation

## Goal

Determine whether the output disagreement between native MXFP8 and BF16
inference corresponds to a meaningful validation-quality difference for the
dense `1e22` checkpoint.

## Setup

The evaluation used `131,072` positions from the 19 available training-data
files dated `2024-04-16`, near the end of the current dataset. Both precision
modes evaluated the same positions in identical batches of `4,096` on a B200.

The checkpoint was:

```text
/artifacts/checkpoints/dense-1e22-native-te-onnx-final.pt
```

The reported uncertainty is the standard error of the paired per-batch
difference. Pairing removes most of the variation in position difficulty that
appears in the standard error of either precision mode alone.

## Command

```bash
uv run eval-precision-modal \
  checkpoints/dense-1e22-native-te-onnx-final.pt \
  --data-glob 'training-run1-test80-20240416-*.tar' \
  --samples 131072 \
  --batch-size 4096
```

## Results

| Metric | Native MXFP8 | Native BF16 | BF16 minus MXFP8 |
| --- | ---: | ---: | ---: |
| Total task loss | 2.820165 | **2.818621** | **-0.001544 +/- 0.000146** |
| Policy loss | 2.001654 | **2.000544** | **-0.001110 +/- 0.000121** |
| Value loss | 0.678060 | **0.678008** | -0.000053 +/- 0.000050 |
| Moves-left loss | 0.140451 | **0.140069** | **-0.000382 +/- 0.000092** |
| Policy top-1 | 52.761% | **52.876%** | **+0.115 +/- 0.048 pp** |

BF16 reduces total loss by approximately `0.055%`. The improvement is small
but measurable and comes primarily from policy and moves-left prediction. Value
loss is effectively tied at this sample size.

## Deployment Context

The corrected `1,000`-position inference comparison measured the following
agreement against native BF16:

| Deployed model | Legal policy top-1 agreement | Policy KL | Q MAE | WDL MAE |
| --- | ---: | ---: | ---: | ---: |
| FP32 ONNX through lc0 `onnx-trt` | **99.2%** | 0.000040 | 0.001234 | 0.000874 |
| FP16 ONNX through lc0 `onnx-trt` | 98.3% | 0.000720 | 0.001353 | 0.005720 |

FP16 policy and Q remain close to BF16, but the evaluation observed rare large
draw-value errors, including a maximum absolute draw error of `0.659`. FP32 is
therefore the safer deployment format.

## Conclusion

MXFP8 inference has a small validation penalty relative to BF16, but the
high-precision deployment path is slightly better rather than worse. There is
no reason to make deployed inference imitate MXFP8 output. Training can retain
MXFP8 for throughput while FP32 ONNX remains the deployment target.
