# Chess Engine 4

Chess Engine 4 is a compact Python training codebase for an LCZero-compatible
neural network. The initial shape is intentionally monolithic: one package owns
data loading, training entrypoints, and eventually model export.

## Prerequisites

- [mise](https://mise.jdx.dev/)

```sh
uv sync --dev
```

## Data

The neural net is trained on lc0 t80 data. Set the data path in `.env`.

## Scripts

`uv run train` currently runs a local data-loading smoke loop. Modal integration
will be added after the local project shape is stable.
