# Dense Inference Mismatch Fix

## Goal

Find and fix the discrepancy between native Transformer Engine inference and
the dense `1e22` ONNX exports running inside lc0.

## Isolation

The initial `1,000`-position evaluation showed only `48.1%` policy top-1
agreement between native TE and the FP32 lc0 export. The evaluation was split
across successive boundaries:

1. native MXFP8 versus native BF16
2. native BF16 versus direct FP32 ONNX on identical plane tensors
3. direct FP32 ONNX versus lc0 at one node

On a `32`-position diagnostic sample, native BF16 and direct FP32 ONNX had
`100%` policy top-1 agreement, policy probability MAE of `3.91e-6`, and `Q` MAE
of `0.00099`. The large discrepancy first appeared at the lc0 input and output
boundary.

## Root Causes

### Rule-50 Input Contract

For `INPUT_CLASSICAL_112_PLANE`, lc0 fills plane `109` with the raw rule-50 ply
count. The Rust training loader divided this value by `99`, so the learned model
saw values in `[0, 1]` while lc0 supplied values in `[0, 99]` to the exported
network.

The fix makes the loader emit the raw lc0 value and moves the division by `99`
inside the dense model. This preserves the semantics learned by existing
checkpoints while making the model's external input contract match lc0. The
normalization is therefore also embedded in every future ONNX export.

### Policy Temperature

lc0 applies a default policy softmax temperature of `1.359` during search. The
original diagnostic compared that adjusted search policy against a raw native
softmax. The inference comparison command now runs lc0 with
`--policy-softmax-temp=1.0` so it measures network equivalence rather than
search calibration.

## Validation

Both corrected exports were regenerated from:

```text
/artifacts/checkpoints/dense-1e22-native-te-onnx-final.pt
```

The final validation used the same `1,000` sampled positions and exact
eight-position histories as the original experiment.

```bash
uv run eval-inference-modal \
  checkpoints/dense-1e22-native-te-onnx-final.pt \
  leela/dense-1e22-native-te-fp32.pb.gz \
  leela/dense-1e22-native-te-fp16.pb.gz \
  --samples 1000 \
  --name dense-1e22-fixed-1k
```

### FP32 ONNX

| Metric | Native BF16 vs lc0 | Native MXFP8 vs lc0 |
| --- | ---: | ---: |
| Policy top-1 agreement | 99.2% | 92.7% |
| Policy KL | 0.000040 | 0.003132 |
| Policy MAE | 0.000183 | 0.001779 |
| Q MAE | 0.001234 | 0.010279 |
| Draw MAE | 0.001064 | 0.005640 |
| WDL MAE | 0.000874 | 0.005515 |

The residual difference between MXFP8 and FP32 lc0 closely tracks the native
MXFP8-versus-BF16 difference and is therefore the expected quantization
boundary rather than an export mismatch.

### FP16 ONNX

FP16 also reached `98.3%` policy top-1 agreement and `Q` MAE of `0.00135`
against native BF16. However, it retained rare large draw-value errors, with a
maximum absolute draw error of `0.659`. FP32 remains the default export format.

The complete corrected result is stored at:

```text
/artifacts/evals/inference-mismatch/dense-1e22-fixed-1k.json
```

## Conclusion

The original train-inference mismatch is fixed. The FP32 lc0 export now closely
matches high-precision native TE inference, and the remaining difference from
the MXFP8 training path is consistent with quantized execution. FP16 is usable
for policy output but remains less reliable for value output.
