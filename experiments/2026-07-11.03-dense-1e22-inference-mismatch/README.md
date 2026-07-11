# Dense 1e22 Inference Mismatch

## Goal

Measure whether the dense `1e22` checkpoint produces the same policy and value
outputs in the native Transformer Engine training pipeline and after export to
lc0's ONNX format. Compare both the FP32 and FP16 ONNX exports against native
MXFP8 inference without investigating the source of any discrepancy.

## Method

The evaluation sampled `1,000` positions from the LCZero training data with
seed `1`. Sampling retained complete game context and skipped the first seven
positions of each game so that lc0 could reconstruct the same eight-position
history consumed during training.

Each position was evaluated in three configurations:

| Configuration | Execution path |
| --- | --- |
| Native reference | Checkpoint loaded into the Transformer Engine model and evaluated with its configured MXFP8 recipe |
| FP32 export | FP32 ONNX weights evaluated by lc0 using `onnx-trt` at one node |
| FP16 export | FP16 ONNX weights evaluated by lc0 using `onnx-trt` at one node |

Policy comparisons use softmax over the legal move indices reported by lc0.
Value comparisons use `Q = P(win) - P(loss)`, draw probability, and the
reconstructed three-class WDL distribution.

## Command

```bash
uv run eval-inference-modal \
  checkpoints/dense-1e22-native-te-onnx-final.pt \
  leela/dense-1e22-native-te-fp32.pb.gz \
  leela/dense-1e22-native-te-fp16.pb.gz \
  --samples 1000 \
  --name dense-1e22-fp32-fp16-1k
```

## Results

All metrics below compare the named ONNX export against native TE inference on
the same positions.

| Metric | FP32 ONNX | FP16 ONNX |
| --- | ---: | ---: |
| Policy top-1 agreement | 48.1% | 47.5% |
| Policy KL, native to export | 0.5931 | 0.5968 |
| Policy MAE | 0.02363 | 0.02368 |
| Policy maximum absolute error | 0.8236 | 0.8282 |
| Q MAE | 0.08982 | 0.09005 |
| Q RMSE | 0.17917 | 0.17938 |
| Q maximum absolute error | 0.96958 | 0.97207 |
| Draw MAE | 0.16777 | 0.16574 |
| Draw RMSE | 0.26900 | 0.25529 |
| Draw maximum absolute error | 0.97394 | 0.94118 |
| WDL MAE | 0.11312 | 0.11819 |
| WDL maximum absolute error | 0.97394 | 0.94118 |

The complete JSON result is stored in the Modal artifact volume at:

```text
/artifacts/evals/inference-mismatch/dense-1e22-fp32-fp16-1k.json
```

## Takeaways

There is a substantial mismatch between native TE inference and both lc0 ONNX
exports. Fewer than half of the positions retain the same top policy move, and
the value outputs also exhibit meaningful average and large worst-case errors.

FP32 and FP16 have nearly identical aggregate policy and Q errors against the
native reference. Their draw and WDL errors differ slightly, but neither export
is consistently close to native inference. Export precision alone therefore
does not appear sufficient to explain the observed mismatch.

This experiment establishes and quantifies the mismatch only. It does not test
possible causes in the input reconstruction, Transformer Engine execution,
ONNX graph, TensorRT backend, or lc0 output interpretation.
