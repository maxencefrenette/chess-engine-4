"""Resumable LCZero archive ingestion into a local Iceberg table."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO
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
MIN_FREE_BYTES = 512 * 1024**3
ConversionResult = tuple[str, int, int, int, list[tuple[str, int, int]], float]


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
    parser.add_argument("--commit-batch-size", type=int, default=16)
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.commit_batch_size <= 0:
        parser.error("--commit-batch-size must be positive")

    warehouse = args.warehouse.resolve()
    warehouse.mkdir(parents=True, exist_ok=True)
    status_path = warehouse / "ingest.status"
    with _exclusive_lock(warehouse / "ingest.lock"):
        try:
            _run_ingestion(
                source=args.source,
                warehouse=warehouse,
                identifier=args.table,
                limit=args.limit,
                workers=args.workers,
                commit_batch_size=args.commit_batch_size,
                status_path=status_path,
            )
        except BaseException as error:
            _write_status(status_path, state="failed", error=repr(error))
            raise


def _run_ingestion(
    *,
    source: Path,
    warehouse: Path,
    identifier: str,
    limit: int | None,
    workers: int,
    commit_batch_size: int,
    status_path: Path,
) -> None:
    archives = sorted(source.glob("*.tar"))
    if limit is not None:
        archives = archives[:limit]
    if not archives:
        raise ValueError(f"no .tar archives found in {source}")

    started = time.perf_counter()
    table = _load_or_create_table(warehouse, identifier)
    data_dir = _local_table_path(table) / "data"
    ledger_path = warehouse / "ingested-archives.txt"
    completed_archives = _read_archive_ledger(ledger_path) | _referenced_archives(table)
    _write_archive_ledger(ledger_path, completed_archives)
    pending = [archive for archive in archives if archive.name not in completed_archives]
    complete_before = len(archives) - len(pending)
    total_records, total_output_bytes = _table_totals(table)
    totals = {
        "archives": complete_before,
        "records": total_records,
        "input_bytes": 0,
        "output_bytes": total_output_bytes,
    }
    _write_status(
        status_path,
        state="running",
        selected=len(archives),
        completed=complete_before,
        pending=len(pending),
        records=total_records,
        output_bytes=total_output_bytes,
    )
    print(
        f"ingestion_plan table={identifier} selected={len(archives)} "
        f"complete={complete_before} pending={len(pending)} workers={workers} "
        f"commit_batch_size={commit_batch_size}",
        flush=True,
    )

    batch: list[ConversionResult] = []
    if workers == 1:
        results = (_convert_archive(archive, data_dir) for archive in pending)
        for result in results:
            batch.append(result)
            if len(batch) == commit_batch_size:
                _register_results(table, batch, totals, len(archives), status_path, ledger_path)
                batch = []
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_convert_archive, archive, data_dir): archive for archive in pending
            }
            for future in concurrent.futures.as_completed(futures):
                batch.append(future.result())
                if len(batch) == commit_batch_size:
                    _register_results(table, batch, totals, len(archives), status_path, ledger_path)
                    batch = []
    if batch:
        _register_results(table, batch, totals, len(archives), status_path, ledger_path)

    elapsed = time.perf_counter() - started
    new_archives = len(pending)
    new_records = totals["records"] - total_records
    new_output_bytes = totals["output_bytes"] - total_output_bytes
    if new_records:
        print(
            f"ingestion_complete archives={new_archives} records={new_records:,} "
            f"input_bytes={totals['input_bytes']:,} total_output_bytes={totals['output_bytes']:,} "
            f"new_output_bytes={new_output_bytes:,} "
            f"compression={new_output_bytes / totals['input_bytes']:.3f} "
            f"bytes_per_row={new_output_bytes / new_records:.2f} "
            f"records_per_sec={new_records / elapsed:,.0f} elapsed_seconds={elapsed:.1f}",
            flush=True,
        )
    else:
        print(f"ingestion_complete archives=0 elapsed_seconds={elapsed:.1f}", flush=True)
    _write_status(
        status_path,
        state="complete",
        selected=len(archives),
        completed=totals["archives"],
        pending=0,
        records=totals["records"],
        output_bytes=totals["output_bytes"],
    )


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


def _referenced_archives(table: Table) -> set[str]:
    if table.current_snapshot() is None:
        return set()
    return {
        f"{Path(urlparse(path).path).stem}.tar"
        for path in table.inspect.files()["file_path"].to_pylist()
    }


def _table_totals(table: Table) -> tuple[int, int]:
    if table.current_snapshot() is None:
        return 0, 0
    files = table.inspect.files()
    return (
        int(sum(files["record_count"].to_pylist())),
        int(sum(files["file_size_in_bytes"].to_pylist())),
    )


def _convert_archive(
    archive: Path,
    data_dir: Path,
) -> ConversionResult:
    free_bytes = shutil.disk_usage(data_dir.parent).free
    if free_bytes < MIN_FREE_BYTES:
        raise OSError(
            f"free-space reserve reached: {free_bytes:,} bytes available, "
            f"{MIN_FREE_BYTES:,} required"
        )
    recovered = _recover_completed_output(archive, data_dir)
    if recovered is not None:
        return recovered
    _remove_orphan_outputs(archive, data_dir)
    started = time.perf_counter()
    records, input_bytes, output_bytes, files = convert_native_lc0_tar_to_iceberg_parquet(
        archive,
        data_dir,
    )
    return archive.name, records, input_bytes, output_bytes, files, time.perf_counter() - started


def _recover_completed_output(archive: Path, data_dir: Path) -> ConversionResult | None:
    partials = list(data_dir.glob(f"num_pieces_trunc=*/{archive.stem}.parquet.partial"))
    paths = list(data_dir.glob(f"num_pieces_trunc=*/{archive.stem}.parquet"))
    if partials or len(paths) != 9:
        return None

    import pyarrow.parquet as pq

    files = [
        (str(path), int(pq.ParquetFile(path).metadata.num_rows), path.stat().st_size)
        for path in sorted(paths)
    ]
    records = sum(file_records for _, file_records, _ in files)
    output_bytes = sum(file_bytes for _, _, file_bytes in files)
    return archive.name, records, archive.stat().st_size, output_bytes, files, -1.0


def _remove_orphan_outputs(archive: Path, data_dir: Path) -> None:
    for path in data_dir.glob(f"num_pieces_trunc=*/{archive.stem}.parquet*"):
        path.unlink()


def _register_results(
    table: Table,
    results: list[ConversionResult],
    totals: dict[str, int],
    selected: int,
    status_path: Path,
    ledger_path: Path,
) -> None:
    paths: list[str] = []
    archives: list[str] = []
    batch_records = 0
    for archive, records, input_bytes, output_bytes, files, elapsed in results:
        file_rows = sum(file_records for _, file_records, _ in files)
        file_bytes = sum(file_bytes for _, _, file_bytes in files)
        if file_rows != records or file_bytes != output_bytes:
            raise ValueError(f"conversion totals do not match partition files for {archive}")
        paths.extend(str(Path(path).resolve()) for path, _, _ in files)
        archives.append(archive)
        batch_records += records
        totals["input_bytes"] += input_bytes
        if elapsed < 0:
            print(
                f"recovered archive={archive} records={records:,} files={len(files)} "
                f"input_bytes={input_bytes:,} output_bytes={output_bytes:,}",
                flush=True,
            )
        elif records:
            print(
                f"converted archive={archive} records={records:,} files={len(files)} "
                f"input_bytes={input_bytes:,} output_bytes={output_bytes:,} "
                f"compression={output_bytes / input_bytes:.3f} "
                f"bytes_per_row={output_bytes / records:.2f} "
                f"records_per_sec={records / elapsed:,.0f} elapsed_seconds={elapsed:.1f}",
                flush=True,
            )
        else:
            print(
                f"converted archive={archive} records=0 files=0 input_bytes={input_bytes:,} "
                f"empty=true elapsed_seconds={elapsed:.1f}",
                flush=True,
            )
    if paths:
        table.add_files(
            paths,
            snapshot_properties={
                "source-archive-count": str(len(archives)),
                "source-archives": ",".join(sorted(archives)),
                "source-records": str(batch_records),
            },
            check_duplicate_files=False,
        )
    _write_archive_ledger(
        ledger_path,
        _read_archive_ledger(ledger_path) | set(archives),
    )
    totals["archives"] += len(results)
    totals["records"] += batch_records
    totals["output_bytes"] += sum(result[3] for result in results)
    print(
        f"committed archives={len(results)} completed={totals['archives']}/{selected} "
        f"records={totals['records']:,} output_bytes={totals['output_bytes']:,}",
        flush=True,
    )
    _write_status(
        status_path,
        state="running",
        selected=selected,
        completed=totals["archives"],
        pending=selected - totals["archives"],
        records=totals["records"],
        output_bytes=totals["output_bytes"],
    )


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[TextIO]:
    handle = path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"another Iceberg ingestion owns {path}") from error
    try:
        yield handle
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def _write_status(path: Path, **values: object) -> None:
    updated = {**values, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    temporary = path.with_suffix(".status.partial")
    temporary.write_text("".join(f"{key}={value}\n" for key, value in updated.items()))
    temporary.replace(path)


def _read_archive_ledger(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line for line in path.read_text().splitlines() if line}


def _write_archive_ledger(path: Path, archives: set[str]) -> None:
    temporary = path.with_suffix(".txt.partial")
    temporary.write_text("".join(f"{archive}\n" for archive in sorted(archives)))
    temporary.replace(path)
