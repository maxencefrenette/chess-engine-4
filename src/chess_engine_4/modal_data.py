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


def audit_parquet_retention_modal() -> None:
    """Count exact deterministic retention capacity without reading Parquet rows."""

    parser = argparse.ArgumentParser(description="Audit retained canonical Parquet rows.")
    parser.add_argument("--batch-size", type=int, default=65_536)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    sources = sorted(_converted_names())
    if not sources:
        parser.error("no canonical Parquet files found on the training-data Volume")
    print(
        f"retention_audit_plan files={len(sources)} batch_size={args.batch_size:,} "
        f"concurrency={_CONVERSION_CONCURRENCY}"
    )
    results: list[dict[str, Any]] = []
    with app.run():
        for result in _audit_retention_one_remote.map(
            sources,
            itertools.repeat(args.batch_size),
            order_outputs=False,
        ):
            results.append(result)
    totals = {
        key: sum(int(result[key]) for result in results)
        for key in (
            "raw_rows",
            "full_rows",
            "half_rows",
            "quarter_rows",
            "usable_full_rows",
            "usable_half_rows",
            "usable_quarter_rows",
        )
    }
    print(
        f"retention_audit_complete files={len(results)} raw_rows={totals['raw_rows']:,} "
        f"full_rows={totals['full_rows']:,} half_rows={totals['half_rows']:,} "
        f"quarter_rows={totals['quarter_rows']:,} "
        f"usable_full_rows={totals['usable_full_rows']:,} "
        f"usable_half_rows={totals['usable_half_rows']:,} "
        f"usable_quarter_rows={totals['usable_quarter_rows']:,}"
    )


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
def _audit_retention_one_remote(source_name: str, batch_size: int) -> dict[str, Any]:
    from chess_engine_4.data.native import native_parquet_retention_counts

    result = native_parquet_retention_counts(
        Path(REMOTE_PARQUET_DATA_PATH) / source_name,
        batch_size=batch_size,
    )
    keys = (
        "raw_rows",
        "full_rows",
        "half_rows",
        "quarter_rows",
        "usable_full_rows",
        "usable_half_rows",
        "usable_quarter_rows",
    )
    return {"file": source_name, **dict(zip(keys, result, strict=True))}


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
