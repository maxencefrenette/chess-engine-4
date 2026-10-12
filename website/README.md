This is the public scaling-laws website for `chess-engine-4`.

The Python package remains at the repository root. This top-level `website/`
folder is a separate Next.js app. Python derives a gitignored JSON payload from
the canonical experiment summaries and model-family recipes. The generator and
scaling fits live in `src/chess_engine_4/analysis/`; no Modal SDK is needed.

Generate that disposable build artifact directly with:

```bash
uv run generate-website-data
```

## Getting Started

After preparing the root Python environment (`uv sync --dev`), install the
website dependencies from this directory:

```bash
mise exec node@24 -- corepack pnpm install --frozen-lockfile
```

Corepack uses the pnpm version in `package.json`. The Node runtime stays managed
by mise; no global package installation is needed. To launch directly from this
directory, use `mise exec node@24 -- corepack pnpm dev`.

From the repository root, launch the development server and scaling-data watcher:

```bash
uv run website
```

The convenience launcher requires `pnpm` on `PATH`.

Open [http://localhost:3000](http://localhost:3000) with your browser.

The development command generates the initial payload and watches its Python,
config, best-runs, and throughput inputs for changes.

The main pages live in `src/app/`. Data loading lives in `src/data/best-runs.ts`.

## Stack

- Next.js App Router
- TypeScript
- Tailwind CSS
- Statically generated scaling-law data

## Build

From `website/`:

```bash
mise exec node@24 -- corepack pnpm lint
mise exec node@24 -- corepack pnpm build
```

The build generates the Python-derived dashboard payload before compiling Next.js.
It requires the root Python dependencies, but no CUDA or native data loader.

## Deploy

Vercel is the intended deployment target once the first public version is ready.
