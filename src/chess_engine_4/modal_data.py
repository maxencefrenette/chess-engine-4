"""Modal commands for converting training data to Parquet."""

from __future__ import annotations

import argparse
import itertools
import os
import time
from pathlib import Path
from typing import Any

from chess_engine_4.modal_train import (
    REMOTE_DATA_PATH,
    REMOTE_PARQUET_DATA_PATH,
    app,
    data_volume,
    image,
)

_CONVERSION_CONCURRENCY = 16


def convert_data_modal() -> None:
    parser = argparse.ArgumentParser(description="Convert LCZero training tar files on Modal.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=8)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--verify-batches", type=int, default=4)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.verify_batches < 0:
        parser.error("--verify-batches must be non-negative")

    sources = _source_names()
    if not sources:
        parser.error("no LCZero tar source files found on the training-data Volume")
    converted = _converted_names()
    pending = [source for source in sources if _output_name(Path(source)) not in converted]
    selected = pending if args.all else pending[: args.limit]
    print(
        f"conversion_plan total={len(sources)} complete={len(converted)} "
        f"selected={len(selected)} concurrency={_CONVERSION_CONCURRENCY}"
    )

    results: list[dict[str, Any]] = []
    with app.run():
        for result in _convert_one_remote.map(selected, order_outputs=False):
            results.append(result)
            print(
                f"converted file={result['file']} records={result['records']:,} "
                f"reduction={result['reduction']:.1%} "
                f"records_per_sec={result['records_per_sec']:,.0f}",
                flush=True,
            )
        verified = (
            _verify_one_remote.remote(sources[0], args.verify_batches)["batches"]
            if args.verify_batches
            else 0
        )
    print(f"conversion_complete files={len(results)} verified_batches={verified}")


def verify_data_modal() -> None:
    parser = argparse.ArgumentParser(description="Verify Parquet files against LCZero tar files.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batches", type=int, default=1)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.batches <= 0:
        parser.error("--batches must be positive")

    sources = _source_names()
    if not sources:
        parser.error("no LCZero tar source files found on the training-data Volume")
    selected = sources if args.limit is None else sources[: args.limit]
    verified_files = 0
    verified_batches = 0
    with app.run():
        for result in _verify_one_remote.map(
            selected,
            itertools.repeat(args.batches),
            order_outputs=False,
        ):
            verified_files += 1
            verified_batches += result["batches"]
            if verified_files % 25 == 0 or verified_files == len(selected):
                print(
                    f"verification_progress files={verified_files}/{len(selected)} "
                    f"batches={verified_batches}",
                    flush=True,
                )
    print(
        f"verification_complete files={verified_files} batches={verified_batches} exact_match=true"
    )


def audit_data_modal() -> None:
    parser = argparse.ArgumentParser(description="Audit live Parquet shard row counts on Modal.")
    parser.parse_args()

    parquet_names = sorted(_converted_names())
    if not parquet_names:
        parser.error("no Parquet files found on the training-data Volume")
    chunk_size = 30
    chunks = [
        parquet_names[offset : offset + chunk_size]
        for offset in range(0, len(parquet_names), chunk_size)
    ]
    results: list[dict[str, float | int]] = []
    with app.run():
        for result in _audit_parquet_metadata_remote.map(chunks, order_outputs=False):
            results.append(result)
            print(
                f"audit_progress chunks={len(results)}/{len(chunks)} "
                f"shards={sum(int(row['shards']) for row in results)}",
                flush=True,
            )
    total_rows = sum(int(result["rows"]) for result in results)
    total_bytes = sum(int(result["bytes"]) for result in results)
    print(
        f"audit_complete shards={sum(int(result['shards']) for result in results)} "
        f"rows={total_rows} bytes={total_bytes} "
        f"bytes_per_row={total_bytes / total_rows:.6f}"
    )


def audit_sources_modal() -> None:
    parser = argparse.ArgumentParser(description="Audit retained LCZero source archives on Modal.")
    parser.parse_args()

    sources = _source_names()
    if not sources:
        parser.error("no LCZero tar source files found on the training-data Volume")
    with app.run():
        result = _audit_sources_remote.remote(sources)
    print(
        f"source_audit_complete archives={result['archives']} games={result['games']} "
        f"rows={result['rows']} duplicate_game_ids={result['duplicate_game_ids']}"
    )


@app.function(
    image=image,
    cpu=2,
    max_containers=_CONVERSION_CONCURRENCY,
    volumes={REMOTE_DATA_PATH: data_volume},
    timeout=30 * 60,
)
def _convert_one_remote(source_name: str) -> dict[str, Any]:
    from chess_engine_4.data.native import convert_native_lc0_tar_to_parquet

    source = Path(REMOTE_DATA_PATH) / source_name
    output_dir = Path(REMOTE_PARQUET_DATA_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / _output_name(source)
    partial = output.with_suffix(".parquet.partial")
    started = time.perf_counter()
    records, input_bytes, output_bytes = convert_native_lc0_tar_to_parquet(source, partial)
    os.replace(partial, output)
    data_volume.commit()
    elapsed = time.perf_counter() - started
    return {
        "file": output.name,
        "records": records,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "reduction": 1.0 - output_bytes / input_bytes,
        "records_per_sec": records / elapsed,
    }


@app.function(
    image=image,
    cpu=2,
    max_containers=_CONVERSION_CONCURRENCY,
    volumes={REMOTE_DATA_PATH: data_volume},
    timeout=10 * 60,
)
def _verify_one_remote(source_name: str, batches: int) -> dict[str, int | str]:
    import torch

    from chess_engine_4.data.native import (
        iter_native_packed_batches,
        iter_native_parquet_batches,
    )

    source = Path(REMOTE_DATA_PATH) / source_name
    output = Path(REMOTE_PARQUET_DATA_PATH) / _output_name(source)
    try:
        tar_batches = iter_native_packed_batches(
            [source], batch_size=256, prefetch_per_thread=1, threads=1
        )
        parquet_batches = iter_native_parquet_batches(
            [output], batch_size=256, prefetch_per_thread=1, threads=1
        )
        verified = 0
        for tar_batch, parquet_batch in zip(tar_batches, parquet_batches, strict=True):
            for tar_tensor, parquet_tensor in zip(tar_batch[:4], parquet_batch[:4], strict=True):
                if not torch.equal(tar_tensor, parquet_tensor):
                    raise ValueError("tar and Parquet training inputs differ")
            if not torch.equal(tar_batch[4][:, 4], parquet_batch[4][:, 4]):
                raise ValueError("tar and Parquet root value targets differ")
            verified += 1
            if verified == batches:
                break
    except Exception as error:
        raise RuntimeError(f"failed to verify {source_name}: {error}") from error
    return {"file": source_name, "batches": verified}


@app.function(
    image=image,
    cpu=2,
    max_containers=_CONVERSION_CONCURRENCY,
    volumes={REMOTE_DATA_PATH: data_volume},
    timeout=30 * 60,
)
def _audit_parquet_metadata_remote(parquet_names: list[str]) -> dict[str, float | int]:
    from chess_engine_4.data.native import native_parquet_row_counts

    paths = [Path(REMOTE_PARQUET_DATA_PATH) / name for name in parquet_names]
    counts = native_parquet_row_counts(paths)
    total_rows = sum(rows for _, rows in counts)
    total_bytes = sum(path.stat().st_size for path in paths)
    if total_rows <= 0:
        raise ValueError("Parquet corpus contains no rows")
    return {
        "shards": len(paths),
        "rows": total_rows,
        "bytes": total_bytes,
        "bytes_per_row": total_bytes / total_rows,
    }


@app.function(
    image=image,
    cpu=2,
    volumes={REMOTE_DATA_PATH: data_volume},
    timeout=30 * 60,
)
def _audit_sources_remote(source_names: list[str]) -> dict[str, int]:
    from chess_engine_4.data.native import inspect_native_lc0_tars

    paths = [Path(REMOTE_DATA_PATH) / name for name in source_names]
    results, duplicate_games = inspect_native_lc0_tars(paths)
    return {
        "archives": len(results),
        "games": sum(games for _, games, _ in results),
        "rows": sum(rows for _, _, rows in results),
        "duplicate_game_ids": duplicate_games,
    }


def _source_names() -> list[str]:
    return sorted(
        entry.path
        for entry in data_volume.listdir("/")
        if entry.type == 1 and entry.path.endswith(".tar")
    )


def _converted_names() -> set[str]:
    return {
        Path(entry.path).name
        for entry in data_volume.listdir("/parquet")
        if entry.type == 1 and entry.path.endswith(".parquet")
    }


def _output_name(source: Path) -> str:
    return f"{source.stem}.parquet"
