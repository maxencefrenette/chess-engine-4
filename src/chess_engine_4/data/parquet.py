"""LCZero-to-Parquet conversion commands."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from chess_engine_4.data.native import convert_native_lc0_tar_to_parquet


def lc0_to_parquet() -> None:
    parser = argparse.ArgumentParser(description="Convert an LCZero v6 tar file to Parquet.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    records, input_bytes, output_bytes = convert_native_lc0_tar_to_parquet(
        args.input,
        args.output,
    )
    elapsed = time.perf_counter() - started
    print(
        f"converted records={records:,} input_bytes={input_bytes:,} "
        f"output_bytes={output_bytes:,} reduction={1 - output_bytes / input_bytes:.1%} "
        f"records_per_sec={records / elapsed:,.0f}"
    )
