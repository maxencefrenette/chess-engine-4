This is the public scaling-laws website for `chess-engine-4`.

The Python package remains at the repository root. This top-level `website/`
folder is a separate Next.js app that reads committed experiment summaries from
`../experiments/best-runs-*.toml`.

## Getting Started

Run the development server:

```bash
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser.

The main pages live in `src/app/`. Data loading lives in `src/data/best-runs.ts`.

## Stack

- Next.js App Router
- TypeScript
- Tailwind CSS
- Static TOML-backed data loading

## Build

```bash
pnpm build
```

## Deploy

Vercel is the intended deployment target once the first public version is ready.
