# MLP 1e21 LC0 Eval

## Goal

Export the dense MLP `1e21` checkpoint to lc0's ONNX weights format and run a
small fastchess sanity check against weak, properly trained lc0 nets.

This is not an Elo estimate. The match sizes are tiny, all games start from the
initial position, and the node odds are intentionally extreme.

## Candidate

Checkpoint:

```text
/artifacts/checkpoints/mlp-1e21-d768x5-b16384-lr2e-4-clip1-moredata-finalckpt-final.pt
```

Exported local weights:

```text
artifacts/leela/mlp-1e21-d768x5-b16384-lr2e-4-clip1-moredata.pb.gz
```

Model notes:

| Item | Value |
| --- | ---: |
| Shape | `d768x5` |
| Parameters | `42.3M` |
| Approx inference FLOPs | `~85M` |
| Training compute budget | `1e21` |
| Training samples | `252.5M` |
| W&B run | https://wandb.ai/maxence-frenette/chess-engine-4/runs/06zpe1og |

## Baselines

| Baseline | Notes |
| --- | --- |
| `t74-744706-128x10` | small legacy LCZero SE CNN, roughly `128x10` |
| `t1-256x10-distilled-swa-2432500` | stronger small attention net, previously too strong |

T74 is cheaper than modern lc0 nets but still likely more expensive per eval
than the dense MLP. A rough parameter-to-FLOPs estimate puts T74 around
`0.35B-0.45B` FLOPs per eval because its convolution weights are reused across
the `8x8` board. The dense MLP is closer to `2 * params`, around `85M` FLOPs.

## Commands

Export:

```bash
uv run modal volume get --force chess-engine-4-artifacts /checkpoints/mlp-1e21-d768x5-b16384-lr2e-4-clip1-moredata-finalckpt-final.pt artifacts/checkpoints/
uv run checkpoint2leela artifacts/checkpoints/mlp-1e21-d768x5-b16384-lr2e-4-clip1-moredata-finalckpt-final.pt --output artifacts/leela/mlp-1e21-d768x5-b16384-lr2e-4-clip1-moredata.pb.gz --onnx-output artifacts/leela/mlp-1e21-d768x5-b16384-lr2e-4-clip1-moredata.onnx
```

T74 eval:

```bash
uv run eval-modal artifacts/leela/mlp-1e21-d768x5-b16384-lr2e-4-clip1-moredata.pb.gz --gpu l4 --candidate-backend onnx-cuda --candidate-nodes 64000 --baseline-nodes 1 --baseline-name t74-744706-128x10 --baseline-weights /artifacts/leela/t74-744706.pb.gz --baseline-url https://storage.lczero.org/files/networks/0df5ca5b7485a043b88bea4f30d5b802033423f2f8ff0ca99b95e0646dd9325c --games 2 --rounds 1 --concurrency 1 --name 1e21-c64000-t74-744706n1-g2-cuda
```

T1 eval:

```bash
uv run eval-modal artifacts/leela/mlp-1e21-d768x5-b16384-lr2e-4-clip1-moredata.pb.gz --gpu l4 --candidate-backend onnx-cuda --candidate-nodes 64000 --baseline-nodes 1 --baseline-name t1-256x10-distilled-swa-2432500 --baseline-weights /artifacts/leela/t1-256x10-distilled-swa-2432500.pb.gz --baseline-url https://storage.lczero.org/files/networks-contrib/t1-256x10-distilled-swa-2432500.pb.gz --games 2 --rounds 1 --concurrency 1 --name 1e21-c64000-t1-256x10n1-g2-cuda
```

## Results

| Match | Result | PGN |
| --- | ---: | --- |
| candidate `64000` nodes vs T74 `1` node | `0.5/2` | `artifacts/evals/1e21-c64000-t74-744706n1-g2-cuda/games.pgn` |
| candidate `64000` nodes vs T1 `1` node | `0/2` | `artifacts/evals/1e21-c64000-t1-256x10n1-g2-cuda/games.pgn` |

The node limit is being honored. The PGNs show candidate moves commonly in the
tens of thousands of nodes, with many moves near the `64k` cap, while the
baseline moves are `n=1`.

## Takeaways

The `1e21` dense MLP is still much weaker than even small, properly trained lc0
nets, but the comparison is not compute-matched or training-compute-matched.
T74 likely spends about `4-5x` more inference FLOPs per node and has orders of
magnitude more training behind it.

The positive signal is that the net is usable inside lc0 through the ONNX
backend and is no longer obviously broken in engine play. It scored `0.5/2`
against T74 at `64000` nodes versus `1` node, while the older `1e16` candidate
scored `0/2` in the same style of test.
