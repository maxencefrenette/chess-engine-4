"""Modal commands for converting training data to Parquet."""

from __future__ import annotations

import argparse
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


def convert_data_modal() -> None:
    parser = argparse.ArgumentParser(description="Convert LCZero training tar files on Modal.")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--verify-batches", type=int, default=4)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.verify_batches < 0:
        parser.error("--verify-batches must be non-negative")

    result: dict[str, Any] | None = None
    with app.run():
        result = _convert_data_remote.remote(args.limit, args.verify_batches)
    if result is None:
        return
    for converted in result["converted"]:
        print(
            f"converted file={converted['file']} records={converted['records']:,} "
            f"reduction={converted['reduction']:.1%} "
            f"records_per_sec={converted['records_per_sec']:,.0f}"
        )
    print(
        f"conversion_complete files={len(result['converted'])} "
        f"verified_batches={result['verified_batches']}"
    )


@app.function(
    image=image,
    cpu=8,
    volumes={REMOTE_DATA_PATH: data_volume},
    timeout=2 * 60 * 60,
)
def _convert_data_remote(limit: int, verify_batches: int) -> dict[str, Any]:
    import torch

    from chess_engine_4.data.native import (
        convert_native_lc0_tar_to_parquet,
        iter_native_packed_batches,
        iter_native_parquet_batches,
    )

    source_dir = Path(REMOTE_DATA_PATH)
    output_dir = Path(REMOTE_PARQUET_DATA_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(source_dir.glob("*.tar"))
    pending = [source for source in sources if not (output_dir / _output_name(source)).exists()]
    converted: list[dict[str, Any]] = []
    converted_pairs: list[tuple[Path, Path]] = []
    for source in pending[:limit]:
        output = output_dir / _output_name(source)
        partial = output.with_suffix(".parquet.partial")
        started = time.perf_counter()
        records, input_bytes, output_bytes = convert_native_lc0_tar_to_parquet(source, partial)
        os.replace(partial, output)
        elapsed = time.perf_counter() - started
        converted.append(
            {
                "file": output.name,
                "records": records,
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
                "reduction": 1.0 - output_bytes / input_bytes,
                "records_per_sec": records / elapsed,
            }
        )
        converted_pairs.append((source, output))
        print(
            f"converted {output.name}: {records:,} records at {records / elapsed:,.0f}/s",
            flush=True,
        )
    data_volume.commit()

    verified = 0
    if verify_batches:
        if converted_pairs:
            source, output = converted_pairs[0]
        else:
            source = next(
                source for source in sources if (output_dir / _output_name(source)).exists()
            )
            output = output_dir / _output_name(source)
        tar_batches = iter_native_packed_batches(
            [source], batch_size=4096, prefetch_per_thread=1, threads=1
        )
        parquet_batches = iter_native_parquet_batches(
            [output], batch_size=4096, prefetch_per_thread=1, threads=1
        )
        for tar_batch, parquet_batch in zip(tar_batches, parquet_batches, strict=True):
            for tar_tensor, parquet_tensor in zip(tar_batch[:4], parquet_batch[:4], strict=True):
                if not torch.equal(tar_tensor, parquet_tensor):
                    raise ValueError("tar and Parquet training inputs differ")
            if not torch.equal(tar_batch[4][:, 4], parquet_batch[4][:, 4]):
                raise ValueError("tar and Parquet root value targets differ")
            verified += 1
            if verified == verify_batches:
                break
    return {"converted": converted, "verified_batches": verified}


def _output_name(source: Path) -> str:
    return f"{source.stem}.parquet"
