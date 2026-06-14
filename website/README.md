This is the public scaling-laws website for `chess-engine-4`.

The Python package remains at the repository root. This top-level `website/`
folder is a separate Next.js app that reads committed experiment summaries from
`../experiments/best-runs-*.toml`.

## Getting Started

Run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser.

The main pages live in `src/app/`. Data loading lives in `src/lib/best-runs.ts`.

## Stack

- Next.js App Router
- TypeScript
- Tailwind CSS
- Static TOML-backed data loading

## Build

```bash
npm run build
```

## Deploy

Vercel is the intended deployment target once the first public version is ready.
