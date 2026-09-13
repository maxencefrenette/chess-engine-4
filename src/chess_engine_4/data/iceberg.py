"""Resumable LCZero archive ingestion into a local Iceberg table."""

from __future__ import annotations

import argparse
import concurrent.futures
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import Table
from pyiceberg.transforms import TruncateTransform
from pyiceberg.types import (
    BinaryType,
    BooleanType,
    FloatType,
    IntegerType,
    NestedField,
    StringType,
)

from chess_engine_4.data.native import convert_native_lc0_tar_to_iceberg_parquet

DEFAULT_NAMESPACE = "training"
DEFAULT_TABLE = "training.t80"
NUM_PIECES_FIELD_ID = 16
NUM_PIECES_PARTITION_FIELD_ID = 1000


def training_position_schema() -> Schema:
    """Return the normalized, Iceberg-compatible LCZero training schema."""

    return Schema(
        NestedField(1, "planes", BinaryType()),
        NestedField(2, "castling", IntegerType()),
        NestedField(3, "side_to_move", BooleanType()),
        NestedField(4, "rule50", IntegerType()),
        NestedField(5, "policy_indices", BinaryType()),
        NestedField(6, "policy_probs_f16", BinaryType()),
        NestedField(7, "root_q", FloatType()),
        NestedField(8, "root_d", FloatType()),
        NestedField(9, "root_m", FloatType()),
        NestedField(10, "best_q", FloatType()),
        NestedField(11, "best_d", FloatType()),
        NestedField(12, "best_m", FloatType()),
        NestedField(13, "result_q", FloatType()),
        NestedField(14, "result_d", FloatType()),
        NestedField(15, "plies_left", FloatType()),
        NestedField(NUM_PIECES_FIELD_ID, "num_pieces", IntegerType()),
        NestedField(17, "source_archive", StringType()),
    )


def training_partition_spec() -> PartitionSpec:
    """Group positions into semantic four-piece ranges."""

    return PartitionSpec(
        PartitionField(
            source_id=NUM_PIECES_FIELD_ID,
            field_id=NUM_PIECES_PARTITION_FIELD_ID,
            transform=TruncateTransform(4),
            name="num_pieces_trunc",
        )
    )


def ingest_iceberg() -> None:
    parser = argparse.ArgumentParser(description="Ingest LCZero tar archives into Iceberg.")
    parser.add_argument("source", type=Path, help="Directory containing completed .tar archives.")
    parser.add_argument("warehouse", type=Path, help="Local Iceberg warehouse directory.")
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    archives = sorted(args.source.glob("*.tar"))
    if args.limit is not None:
        archives = archives[: args.limit]
    if not archives:
        parser.error(f"no .tar archives found in {args.source}")

    started = time.perf_counter()
    table = _load_or_create_table(args.warehouse, args.table)
    data_dir = _local_table_path(table) / "data"
    referenced = _referenced_files(table)
    pending = [archive for archive in archives if not _archive_is_referenced(archive, referenced)]
    print(
        f"ingestion_plan table={args.table} selected={len(archives)} "
        f"complete={len(archives) - len(pending)} pending={len(pending)} workers={args.workers}"
    )

    totals = {"records": 0, "input_bytes": 0, "output_bytes": 0}
    if args.workers == 1:
        results = (_convert_archive(archive, data_dir) for archive in pending)
        for result in results:
            _register_result(table, result, totals)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_convert_archive, archive, data_dir): archive for archive in pending
            }
            for future in concurrent.futures.as_completed(futures):
                _register_result(table, future.result(), totals)

    elapsed = time.perf_counter() - started
    records = totals["records"]
    if records:
        print(
            f"ingestion_complete archives={len(pending)} records={records:,} "
            f"input_bytes={totals['input_bytes']:,} output_bytes={totals['output_bytes']:,} "
            f"compression={totals['output_bytes'] / totals['input_bytes']:.3f} "
            f"bytes_per_row={totals['output_bytes'] / records:.2f} "
            f"records_per_sec={records / elapsed:,.0f} elapsed_seconds={elapsed:.1f}"
        )
    else:
        print(f"ingestion_complete archives=0 elapsed_seconds={elapsed:.1f}")


def _load_or_create_table(warehouse: Path, identifier: str) -> Table:
    warehouse = warehouse.resolve()
    warehouse.mkdir(parents=True, exist_ok=True)
    catalog = SqlCatalog(
        "local",
        uri=f"sqlite:///{warehouse / 'catalog.db'}",
        warehouse=warehouse.as_uri(),
    )
    namespace = identifier.rsplit(".", 1)[0] if "." in identifier else DEFAULT_NAMESPACE
    catalog.create_namespace_if_not_exists(namespace)
    table = catalog.create_table_if_not_exists(
        identifier,
        schema=training_position_schema(),
        partition_spec=training_partition_spec(),
        properties={
            "format-version": "2",
            "write.parquet.compression-codec": "zstd",
            "write.target-file-size-bytes": str(512 * 1024 * 1024),
        },
    )
    if table.schema() != training_position_schema():
        raise ValueError(f"existing table {identifier} has a different schema")
    if table.spec() != training_partition_spec():
        raise ValueError(f"existing table {identifier} has a different partition spec")
    return table


def _local_table_path(table: Table) -> Path:
    parsed = urlparse(table.location())
    if parsed.scheme != "file":
        raise ValueError(f"only local file warehouses are supported, got {table.location()}")
    return Path(unquote(parsed.path))


def _referenced_files(table: Table) -> set[str]:
    if table.current_snapshot() is None:
        return set()
    return set(table.inspect.files()["file_path"].to_pylist())


def _archive_is_referenced(archive: Path, referenced: set[str]) -> bool:
    expected_name = f"{archive.stem}.parquet"
    return any(Path(urlparse(path).path).name == expected_name for path in referenced)


def _convert_archive(
    archive: Path,
    data_dir: Path,
) -> tuple[str, int, int, int, list[tuple[str, int, int]], float]:
    _remove_orphan_outputs(archive, data_dir)
    started = time.perf_counter()
    records, input_bytes, output_bytes, files = convert_native_lc0_tar_to_iceberg_parquet(
        archive,
        data_dir,
    )
    return archive.name, records, input_bytes, output_bytes, files, time.perf_counter() - started


def _remove_orphan_outputs(archive: Path, data_dir: Path) -> None:
    for path in data_dir.glob(f"num_pieces_trunc=*/{archive.stem}.parquet*"):
        path.unlink()


def _register_result(
    table: Table,
    result: tuple[str, int, int, int, list[tuple[str, int, int]], float],
    totals: dict[str, int],
) -> None:
    archive, records, input_bytes, output_bytes, files, elapsed = result
    file_rows = sum(file_records for _, file_records, _ in files)
    file_bytes = sum(file_bytes for _, _, file_bytes in files)
    if file_rows != records or file_bytes != output_bytes:
        raise ValueError(f"conversion totals do not match partition files for {archive}")
    table.add_files(
        [str(Path(path).resolve()) for path, _, _ in files],
        snapshot_properties={
            "source-archive": archive,
            "source-records": str(records),
        },
    )
    totals["records"] += records
    totals["input_bytes"] += input_bytes
    totals["output_bytes"] += output_bytes
    print(
        f"ingested archive={archive} records={records:,} files={len(files)} "
        f"input_bytes={input_bytes:,} output_bytes={output_bytes:,} "
        f"compression={output_bytes / input_bytes:.3f} "
        f"bytes_per_row={output_bytes / records:.2f} "
        f"records_per_sec={records / elapsed:,.0f} elapsed_seconds={elapsed:.1f}"
    )
