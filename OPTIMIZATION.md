# Optimization Notes

Experiment and promotion rules are maintained in [AGENTS.md](AGENTS.md#experiments).
Infrastructure settings should only reduce cost or wall-clock time for an
otherwise identical run. Research CLI usage is in [docs/research.md](docs/research.md).

These implementation choices were established by the recorded cloud experiments.
They remain unchanged by the local-first package reorganization; they are not
new throughput measurements or VRAM-fit guarantees for the local GPU.

`compare-run` reports `EG_flops`: fitted training FLOPs required to reach the
observed loss divided by the run's actual FLOPs. Values above `1x` beat the
fitted frontier.

## Transformer Engine

Training is CUDA-only and uses NVIDIA Transformer Engine for MLP layers. Dense
models use TE's high-level CUDA graph API.

The canonical dense recipe uses RTX PRO 6000 through d256 and B200 from d512
upward, based on measured Modal cost per step. The MoE recipe uses custom kernels
on RTX PRO 6000 at d128 and d256, then Transformer Engine on B200. A100 is
supported for BF16 experiments but is not canonical because measured training
cost is higher. `infra.gpu` records this infrastructure choice; `--gpu` is an
experiment override, not an automatic fallback.

The `moe64a2` family alternates dense layers with Transformer Engine's fused
top-k router and grouped linear kernels. Widths below 1024 use static
CUDA-graphed dispatch; larger widths use TE's faster fused permutation and
unpermutation path. Total parameters include all 64 experts in each MoE layer;
physical training FLOPs include only the two active experts plus dense and shared
layers. Alternation is a fixed part of the family while small grouped MXFP8
kernels remain expensive.

`infra.cpu_cores` controls the physical CPU cores reserved by Modal, while
`infra.dataloader_threads` controls the Rust batch-loading workers. Baseline
configs reserve eight cores and use eight workers; larger runs can reserve more
CPU headroom without changing the data-loading topology.

Dense d64 copies directly from pageable memory because explicit pinning costs
more than it saves for its smaller batch. Dense d128 and d256 reuse
two pinned host staging slots to avoid per-step pinned allocations. Dense d512
and larger overlap H2D transfer with the previous step on a dedicated CUDA
stream. These paths are deliberately shape-specific: paired same-container
benchmarks found that staging regressed at d512, while copy-stream coordination
did not pay for itself below that width. Additional Modal CPUs and loader workers
did not improve end-to-end throughput, so dense configs continue to use eight of
each.

The selected path is explicit in each model recipe as `model.input_pipeline`;
the training loop does not infer it from model family or width.
