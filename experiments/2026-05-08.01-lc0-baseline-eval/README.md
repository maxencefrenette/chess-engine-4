# LC0 Baseline Eval

## Goal

Convert the current `1e16` MLP checkpoint into an lc0-compatible ONNX weights file and run early strength sanity checks against existing LCZero networks.

The candidate was the `1e16-baseline-checkpoint-final.pt` checkpoint exported to `artifacts/leela/1e16-baseline.pb.gz`.

## Candidate

The exported candidate is the current `1e16` MLP-only model:

- Shape: `d160x6`
- Total params: about `3.2M`
- Approx inference FLOPs: about `6.6M` per network eval
- Training samples: about `55.5M`
- Training target: result-supervised WDL plus policy and moves-left heads

## Baselines

| Baseline | Architecture | Size notes |
| --- | --- | --- |
| `BT4-1740` | large attention body | far too strong for a first benchmark |
| `t3-512x15x16h-distill-swa-2767500` | attention body, `512x15` | much larger than candidate |
| `t1-256x10-distilled-swa-2432500` | attention body, `256x10` | about `20.2M` params, `35 MB` compressed |
| `t74-744706-128x10` | SE CNN, `128x10` | small legacy LCZero net, `6.1 MB` compressed |

The T74 net used here is run-2 network `744706`, SHA `0df5ca5b7485a043b88bea4f30d5b802033423f2f8ff0ca99b95e0646dd9325c`.

## Results

| Match | Result |
| --- | --- |
| candidate `1000` nodes vs BT4 `1` node | `0/2` |
| candidate `4000` nodes vs BT4 `1` node | `0/2` |
| candidate `16000` nodes vs BT4 `1` node | `0/2` |
| candidate `16000` nodes vs T3 `1` node | `0/2` |
| candidate `64000` nodes vs T3 `1` node | `0/2` |
| candidate `64000` nodes vs T1 `1` node | `0/2` |
| candidate `64000` nodes vs T74 `1` node | `0/2` |

PGNs included in this folder:

- `t1-256x10-games.pgn`
- `t74-744706-games.pgn`

Other PGNs remain in the Modal artifact volume under `/artifacts/evals/...`.

## Runtime Checks

The node limit is being honored. After adding PGN telemetry, fastchess reports candidate move counts around the configured node target. For example, the `64k` node matches show candidate moves commonly in the `30k-64k` node range, with many near the cap.

The lc0 build was also fixed to use `-Dnative_arch=false`. The earlier binary had been built with host-specific CPU instructions and could crash on different Modal workers.

## Interpretation

The candidate is not obviously broken. It develops pieces, contests the center, converts trivial material-imbalance positions into winning/losing WDL values, and often keeps its own evaluation near equality for a while.

The main failure mode is value blindness. The candidate often blunders a piece and then fails to recognize that the resulting position is lost. It's as if the net is blind to losing some pieces due to insufficient generalization from the limited training data.

## Next Steps

Use smaller or more targeted evals before treating full-engine Elo as the main metric:

- evaluate tactical FEN suites with candidate WDL at `1` node
- compare candidate and LCZero baseline evals on positions from these PGNs
- consider training value against search `root_q/root_d` to get a better value signal
