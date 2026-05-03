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

The simplified Leela reader expects plain `.tar` files whose members contain
packed LCZero v5/v6 binary records. Members may also be gzip-compressed.

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
