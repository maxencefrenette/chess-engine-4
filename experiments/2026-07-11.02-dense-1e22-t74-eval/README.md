# Dense 1e22 vs T74 Eval

## Goal

Evaluate the FP32 export of the dense `1e22` net against the small, properly
trained T74 LCZero net. This is a strength sanity check, not a compute-matched
comparison: the candidate receives `100` nodes per move and T74 receives `1`.

## Setup

| Item | Candidate | Baseline |
| --- | --- | --- |
| Network | dense `1e22` FP32 export | `t74-744706-128x10` |
| Backend | `onnx-trt` | `cuda` |
| Nodes | `100` | `1` |

The match used fastchess with the hardcoded Stockfish `noob_2moves.epd`
opening book. It played `100` paired openings with colors reversed, for `200`
total games.

## Command

```bash
uv run eval-modal artifacts/leela/dense-1e22-native-te-fp32.pb.gz \
  --candidate-name dense-1e22-fp32 \
  --candidate-nodes 100 \
  --candidate-backend onnx-trt \
  --baseline-name t74-744706-128x10 \
  --baseline-nodes 1 \
  --baseline-backend cuda \
  --baseline-weights /artifacts/leela/t74-744706.pb.gz \
  --baseline-url https://storage.lczero.org/files/networks/0df5ca5b7485a043b88bea4f30d5b802033423f2f8ff0ca99b95e0646dd9325c \
  --games 2 \
  --rounds 100 \
  --concurrency 1 \
  --startup-ms 600000 \
  --ping-ms 600000 \
  --name dense-1e22-fp32-n100-vs-t74-n1-noob2-p100
```

## Results

| Candidate wins | Draws | Candidate losses | Score | Estimated Elo difference |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 6 | 194 | 1.5% | -727 |

All `200` games terminated normally. The PGN and fastchess log are stored in
Modal under:

```text
/artifacts/evals/dense-1e22-fp32-n100-vs-t74-n1-noob2-p100/
```

They were also downloaded locally to the corresponding path under
`artifacts/evals/`.

## Interpretation

At these node odds, the dense `1e22` net remains decisively weaker than T74.
This does not isolate network quality per inference FLOP because the match is
deliberately not compute-matched, and T74 has substantially more training
compute behind it.

The `100` node value is the configured lc0 search limit. Batched search may
overshoot it within a search iteration; the log contains candidate searches
that report more than exactly `100` visited nodes. T74 consistently receives
the configured `1` node.
