# Development and verification

The project owns its Python/Rust toolchains through mise and its packages through
uv. Keep uv on `latest`. The website owns its dependencies separately through pnpm.
Do not install project packages globally or change the Porygon image for setup.

```sh
mise install
uv sync --dev
mise run setup-native
```

`setup-native` builds the Rust data loader for CPU data tests. Rebuild after
changing the Rust loader or Python interpreter. It does not require CUDA or
Transformer Engine. Compilation can use substantial CPU; on a busy host use
`CARGO_BUILD_JOBS=1 mise run setup-native`, or defer it until capacity is available.

For local CUDA training, use `mise run setup-cuda`; this installs the CUDA extra
and builds the native loader. Do not run it just to edit docs or generate the site.
For cloud adapter work, use the separate [Modal extra](cloud.md).

## Verification

Fast checks without tests, native compilation, or website builds:

```sh
mise run check-fast
```

After native setup, `mise run test` runs the base CPU suite. Modal-specific test
modules skip if its SDK is absent. `uv run --extra modal pytest -q` includes them;
neither suite launches training runs, benchmarks, or cloud jobs.

The repository gate is in [AGENTS.md](../AGENTS.md#verification). For changes to
package boundaries/cloud adapters, check both base and Modal-extra environments.
`tests/test_local_workflow.py` also blocks Modal imports in a fresh interpreter
to catch accidental dependencies even when the SDK is installed.

Keep verification sequential at low priority on a shared workstation. Limit CPU
affinity and OpenMP/BLAS/Rayon threads. For a website build on a busy host,
`CIRCLE_NODE_TOTAL=1` limits Next.js page workers. CUDA numerical checks and
performance benchmarks are separate explicit jobs.

## Website

With the root Python environment prepared, install the website's pinned pnpm
dependencies using a mise-managed Node 24:

```sh
cd website
mise exec node@24 -- corepack pnpm install --frozen-lockfile
mise exec node@24 -- corepack pnpm lint
mise exec node@24 -- corepack pnpm build
```

Use `corepack pnpm dev` under the same mise command to start the development
server. From the repository root, `uv run website` is a convenience launcher
when `pnpm` is already on `PATH`. The build and dev commands generate the
Python-derived dashboard data; neither requires CUDA, Modal, or the Rust loader.
See [website/README.md](../website/README.md).

## Engine and kernels

Keep `third_party/lc0` and project-owned inference kernels: they implement the
release target. `uv run build-lc0` compiles the engine and inference kernels;
it is not a lightweight check. Build requirements and kernel policy are in
[kernels/README.md](../kernels/README.md).

The lc0 fork is maintained as a git subtree. When deliberately updating it:

```sh
git subtree pull --prefix=third_party/lc0 https://github.com/LeelaChessZero/lc0.git master --squash
```
