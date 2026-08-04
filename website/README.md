This is the public scaling-laws website for `chess-engine-4`.

The Python package remains at the repository root. This top-level `website/`
folder is a separate Next.js app. Python derives a gitignored JSON payload from
the canonical experiment summaries and model-family recipes.

## Getting Started

Run the development server:

```bash
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser.

The development command generates the initial payload and watches its Python,
config, and best-runs inputs for changes.

The main pages live in `src/app/`. Data loading lives in `src/data/best-runs.ts`.

## Stack

- Next.js App Router
- TypeScript
- Tailwind CSS
- Statically generated scaling-law data

## Build

```bash
pnpm build
```

## Deploy

Vercel is the intended deployment target once the first public version is ready.
