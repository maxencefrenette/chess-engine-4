"""Resumably download the complete LCZero t80 archive inventory."""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import shutil
import signal
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import BinaryIO

BASE_URL = "https://data.lczero.org/files/training_data/test80"
DEFAULT_DATA_ROOT = Path("/data/chess/t80")
DEFAULT_RATE_LIMIT_MIB = 64
DEFAULT_RESERVE_GIB = 512
CHUNK_SIZE = 1024 * 1024
RETRY_DELAY_SECONDS = 5 * 60
ARCHIVE_DELAY_SECONDS = 5
USER_AGENT = "Mozilla/5.0 chess-engine-4-desktop-sync"

type InventoryEntry = tuple[str, int]

_INVENTORY_PATTERN = re.compile(
    rb'href="(training-run1-test80-\d{8}-\d{4}\.tar)".*?\s(\d+)\s*$',
    re.MULTILINE,
)


def parse_inventory_html(content: bytes) -> list[InventoryEntry]:
    """Parse archive names and advertised byte sizes from the upstream index."""

    return [(name.decode(), int(size)) for name, size in _INVENTORY_PATTERN.findall(content)]


def read_inventory(path: Path) -> list[InventoryEntry]:
    entries: list[InventoryEntry] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            name, size = line.split("\t", maxsplit=1)
            advertised_size = int(size)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid inventory row") from exc
        if not re.fullmatch(r"training-run1-test80-\d{8}-\d{4}\.tar", name):
            raise ValueError(f"{path}:{line_number}: invalid archive name {name!r}")
        if advertised_size <= 0:
            raise ValueError(f"{path}:{line_number}: archive size must be positive")
        entries.append((name, advertised_size))
    if not entries:
        raise ValueError(f"{path}: inventory is empty")
    return entries


def response_write_mode(*, offset: int, status: int, content_range: str | None) -> str:
    """Select safe partial-file write mode for an HTTP range response."""

    if offset == 0:
        return "wb"
    if status == 200:
        return "wb"
    if status != 206:
        raise RuntimeError(f"resume request returned HTTP {status}, expected 206")
    expected_prefix = f"bytes {offset}-"
    if content_range is None or not content_range.startswith(expected_prefix):
        raise RuntimeError(
            f"resume response has Content-Range {content_range!r}, expected {expected_prefix!r}"
        )
    return "ab"


class Downloader:
    def __init__(
        self,
        *,
        data_root: Path,
        base_url: str,
        rate_limit_mib: int,
        reserve_gib: int,
    ) -> None:
        self.data_root = data_root
        self.source_dir = data_root / "source"
        self.inventory_path = data_root / "inventory.tsv"
        self.listing_path = data_root / "upstream-index.html"
        self.log_path = data_root / "download.log"
        self.status_path = data_root / "status"
        self.lock_path = data_root / "download.lock"
        self.base_url = base_url.rstrip("/")
        self.rate_limit_bytes = rate_limit_mib * 1024 * 1024
        self.reserve_bytes = reserve_gib * 1024**3
        self.stop_requested = False
        self.completed_files = 0
        self.completed_bytes = 0
        self.current_file = ""
        self._lock_handle: BinaryIO | None = None

    def run(self) -> None:
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self._acquire_lock()
        self._install_signal_handlers()
        inventory = self._load_or_freeze_inventory()
        total_files = len(inventory)
        total_bytes = sum(size for _, size in inventory)
        self.completed_files, self.completed_bytes = self._scan_completed(inventory)
        self._log(
            f"start files={total_files} bytes={total_bytes} "
            f"already_complete={self.completed_files} "
            f"rate_limit_mib={self.rate_limit_bytes // 1024**2}"
        )
        self._write_status("running")

        for name, advertised_size in inventory:
            if self.stop_requested:
                break
            destination = self.source_dir / name
            if destination.exists():
                actual_size = destination.stat().st_size
                if actual_size != advertised_size:
                    self._fail(
                        "invalid_complete",
                        name=name,
                        expected=advertised_size,
                        actual=actual_size,
                    )
                continue
            self._download_archive(name, advertised_size)
            if self.stop_requested:
                break
            self.completed_files += 1
            self.completed_bytes += advertised_size
            self._log(
                f"complete file={name} files={self.completed_files}/{total_files} "
                f"bytes={self.completed_bytes}/{total_bytes}"
            )
            self.current_file = ""
            self._write_status("running")
            self._interruptible_sleep(ARCHIVE_DELAY_SECONDS)

        if self.stop_requested:
            self._log(f"paused file={self.current_file}")
            self._write_status("paused")
        else:
            self._log(
                f"finished files={self.completed_files} bytes={self.completed_bytes}"
            )
            self._write_status("complete")

    def _download_archive(self, name: str, advertised_size: int) -> None:
        partial = self.source_dir / f"{name}.partial"
        if partial.exists() and partial.stat().st_size > advertised_size:
            self._fail(
                "oversized_partial",
                name=name,
                expected=advertised_size,
                actual=partial.stat().st_size,
            )
        available_bytes = shutil.disk_usage(self.data_root).free
        if available_bytes < advertised_size + self.reserve_bytes:
            self._fail(
                "space_guard",
                name=name,
                available=available_bytes,
                reserve=self.reserve_bytes,
            )

        self.current_file = name
        self._write_status("running")
        attempt = 0
        while not self.stop_requested:
            attempt += 1
            self._log(f"download file={name} bytes={advertised_size} attempt={attempt}")
            try:
                self._transfer(name, partial)
                actual_size = partial.stat().st_size
                if actual_size == advertised_size:
                    partial.replace(self.source_dir / name)
                    return
                if actual_size > advertised_size:
                    self._fail(
                        "oversized_partial",
                        name=name,
                        expected=advertised_size,
                        actual=actual_size,
                    )
                self._log(
                    f"size_mismatch file={name} expected={advertised_size} actual={actual_size}"
                )
            except (OSError, RuntimeError, urllib.error.URLError) as exc:
                self._log(f"transfer_failed file={name} attempt={attempt} error={exc!r}")
            self._interruptible_sleep(RETRY_DELAY_SECONDS)

    def _transfer(self, name: str, partial: Path) -> None:
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(
            f"{self.base_url}/{name}",
            headers={
                "User-Agent": USER_AGENT,
                **({"Range": f"bytes={offset}-"} if offset else {}),
            },
        )
        started_at = time.monotonic()
        transferred = 0
        with urllib.request.urlopen(request, timeout=300) as response:
            mode = response_write_mode(
                offset=offset,
                status=response.status,
                content_range=response.headers.get("Content-Range"),
            )
            with partial.open(mode) as handle:
                while not self.stop_requested:
                    block = response.read(CHUNK_SIZE)
                    if not block:
                        break
                    handle.write(block)
                    transferred += len(block)
                    target_elapsed = transferred / self.rate_limit_bytes
                    delay = target_elapsed - (time.monotonic() - started_at)
                    if delay > 0:
                        time.sleep(delay)

    def _load_or_freeze_inventory(self) -> list[InventoryEntry]:
        if self.inventory_path.exists() and self.inventory_path.stat().st_size:
            return read_inventory(self.inventory_path)
        request = urllib.request.Request(
            f"{self.base_url}/",
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            content = response.read()
        inventory = parse_inventory_html(content)
        if not inventory:
            raise RuntimeError("no archives parsed from upstream listing")
        self._atomic_write(self.listing_path, content)
        inventory_content = "".join(f"{name}\t{size}\n" for name, size in inventory).encode()
        self._atomic_write(self.inventory_path, inventory_content)
        return inventory

    def _scan_completed(self, inventory: Iterable[InventoryEntry]) -> tuple[int, int]:
        completed_files = 0
        completed_bytes = 0
        for name, advertised_size in inventory:
            destination = self.source_dir / name
            if destination.exists() and destination.stat().st_size == advertised_size:
                completed_files += 1
                completed_bytes += advertised_size
        return completed_files, completed_bytes

    def _acquire_lock(self) -> None:
        self._lock_handle = self.lock_path.open("ab")
        try:
            fcntl.flock(self._lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another t80 downloader holds {self.lock_path}") from None

    def _install_signal_handlers(self) -> None:
        def request_stop(_signum: int, _frame: FrameType | None) -> None:
            self.stop_requested = True

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)

    def _interruptible_sleep(self, seconds: int) -> None:
        deadline = time.monotonic() + seconds
        while not self.stop_requested and (remaining := deadline - time.monotonic()) > 0:
            time.sleep(min(remaining, 1))

    def _fail(self, event: str, **values: object) -> None:
        details = " ".join(f"{key}={value}" for key, value in values.items())
        self._log(f"{event} {details}")
        self._write_status("error")
        raise RuntimeError(f"{event}: {details}")

    def _log(self, message: str) -> None:
        line = f"{datetime.now(UTC).isoformat()} {message}"
        print(line, flush=True)
        with self.log_path.open("a") as handle:
            handle.write(f"{line}\n")

    def _write_status(self, state: str) -> None:
        content = (
            f"state={state}\n"
            f"completed_files={self.completed_files}\n"
            f"completed_bytes={self.completed_bytes}\n"
            f"current_file={self.current_file}\n"
            f"updated_at={datetime.now(UTC).isoformat()}\n"
            f"pid={os.getpid() if state == 'running' else ''}\n"
        ).encode()
        self._atomic_write(self.status_path, content)

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--rate-limit-mib", type=int, default=DEFAULT_RATE_LIMIT_MIB)
    parser.add_argument("--reserve-gib", type=int, default=DEFAULT_RESERVE_GIB)
    args = parser.parse_args()
    if args.rate_limit_mib <= 0:
        parser.error("--rate-limit-mib must be positive")
    if args.reserve_gib < 0:
        parser.error("--reserve-gib must be non-negative")
    downloader = Downloader(
        data_root=args.data_root,
        base_url=args.base_url,
        rate_limit_mib=args.rate_limit_mib,
        reserve_gib=args.reserve_gib,
    )
    try:
        downloader.run()
    except (OSError, RuntimeError, ValueError, urllib.error.URLError) as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
