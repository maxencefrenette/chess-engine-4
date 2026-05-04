# Chess Engine 4

Chess Engine 4 is a compact Python training codebase for an LCZero-compatible
neural network. The initial shape is intentionally monolithic: one package owns
data loading, training entrypoints, and eventually model export.

## Tooling

- Python: 3.14
- Package manager: uv
- Tool manager: mise
- Training stack: PyTorch

```sh
mise install
uv sync --dev
```

## Data

The Leela reader expects plain `.tar` files whose members contain gzip-compressed
LCZero v6 binary chunks. Each dataset item is already a PyTorch training batch
with LCZero-shaped tensors:

- `planes`: `[batch, 112, 8, 8]`
- `policy`: `[batch, 1858]`, preserving `-1` illegal-move sentinels
- `value`: `[batch, 6, 3]` for result, best, played, orig, root, and
  short-term targets

Set the data path with:

```sh
export CHESS_ENGINE_4_DATA_PATH=/path/to/leela-data
```

The value can be a `.tar` file, a directory of `.tar` files, a glob, or multiple
entries separated by the OS path separator.

## Scripts

```sh
uv run inspect-data
uv run sample-batch
uv run train
```

`uv run train` currently runs a local data-loading smoke loop. Modal integration
will be added after the local project shape is stable.
