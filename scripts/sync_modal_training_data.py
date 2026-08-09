"""Sync LCZero training tar files directly into the Modal training-data Volume."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path

import modal

APP_NAME = "chess-engine-4-data-sync"
DATA_VOLUME_NAME = "chess-engine-4-training-data"
REMOTE_DATA_PATH = "/data/training_data"
BASE_URL = "https://data.lczero.org/files/training_data/test80"
DEFAULT_MINIMUM_SOURCE_MIB = 100

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)


def select_source_candidates(
    candidates: list[tuple[str, int]], file_count: int
) -> list[tuple[str, int]]:
    selected = candidates[:file_count]
    if len(selected) != file_count:
        raise RuntimeError(
            f"requested {file_count} files, but only {len(selected)} candidates are available"
        )
    return selected


def unexpected_complete_sources(
    current_sources: set[str], retained_sources: set[str], selected_names: set[str]
) -> set[str]:
    return current_sources - retained_sources - selected_names


def validate_sync_run_inventory(
    current_sources: set[str], retained_sources: set[str], selected_names: set[str]
) -> None:
    unexpected_sources = unexpected_complete_sources(
        current_sources, retained_sources, selected_names
    )
    if unexpected_sources:
        raise RuntimeError(
            "unexpected complete sources appeared during acquisition: "
            f"{sorted(unexpected_sources)}"
        )


def resolve_retained_sources_for_run(
    manifest_retained_sources: list[str],
    invocation_retained_sources: list[str],
    *,
    resume_existing_run: bool,
) -> set[str]:
    if resume_existing_run:
        return set(manifest_retained_sources)
    if manifest_retained_sources != invocation_retained_sources:
        raise RuntimeError("sync-run retained-source inventory does not match this invocation")
    return set(invocation_retained_sources)


@app.function(
    image=modal.Image.debian_slim(python_version="3.14"),
    volumes={REMOTE_DATA_PATH: volume},
    timeout=24 * 60 * 60,
)
def sync_files(
    start_day: str,
    file_count: int,
    minimum_source_bytes: int,
    dry_run: bool,
    download_concurrency: int,
    expected_retained_sources: list[str],
    sync_run_id: str,
    resume_existing_run: bool,
) -> list[dict[str, int | str]]:
    import os
    import re
    import shutil
    import time
    import urllib.error
    import urllib.request
    from html.parser import HTMLParser

    os.makedirs(REMOTE_DATA_PATH, exist_ok=True)

    class LinkParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.hrefs: list[str] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag != "a":
                return
            href = dict(attrs).get("href")
            if href is not None:
                self.hrefs.append(href)

    listing_request = urllib.request.Request(
        f"{BASE_URL}/",
        headers={"User-Agent": "Mozilla/5.0 chess-engine-4-data-sync"},
    )
    with urllib.request.urlopen(listing_request) as response:
        parser = LinkParser()
        listing_html = response.read().decode("utf-8")
        parser.feed(listing_html)

    filename_pattern = re.compile(r"training-run1-test80-(\d{8})-\d{4}\.tar$")
    listing_pattern = re.compile(
        r'href="(training-run1-test80-\d{8}-\d{4}\.tar)"[^\n]*?\s(\d+)\s*$',
        re.MULTILINE,
    )
    advertised_sizes = {name: int(size) for name, size in listing_pattern.findall(listing_html)}
    converted = {
        Path(entry.path).stem
        for entry in volume.listdir("/parquet")
        if entry.type == 1 and entry.path.endswith(".parquet")
    }
    current_sources = {
        entry.path
        for entry in volume.listdir("/")
        if entry.type == 1 and entry.path.endswith(".tar")
    }
    retained_sources = set(expected_retained_sources)
    sync_run_dir = Path(REMOTE_DATA_PATH) / "source-manifests" / "sync-runs"
    sync_run_path = sync_run_dir / f"{sync_run_id}.json"
    candidates = sorted(
        (name, advertised_sizes[name])
        for href in parser.hrefs
        if (match := filename_pattern.fullmatch(name := href.rsplit("/", 1)[-1]))
        and match.group(1) >= start_day
        and name.removesuffix(".tar") not in converted
        and name not in retained_sources
        and advertised_sizes.get(name, 0) >= minimum_source_bytes
    )
    if sync_run_path.exists():
        sync_run = json.loads(sync_run_path.read_text())
        retained_sources = resolve_retained_sources_for_run(
            sync_run["expected_retained_sources"],
            expected_retained_sources,
            resume_existing_run=resume_existing_run,
        )
        selected = [(row["name"], row["bytes"]) for row in sync_run["selected"]]
        if len(selected) != file_count:
            raise RuntimeError("sync-run selected count does not match this invocation")
    else:
        if resume_existing_run:
            raise RuntimeError(f"sync-run manifest does not exist: {sync_run_id}")
        if current_sources != retained_sources:
            raise RuntimeError(
                "source inventory changed between local preflight and remote selection: "
                f"expected={sorted(retained_sources)} actual={sorted(current_sources)}"
            )
        try:
            selected = select_source_candidates(candidates, file_count)
        except RuntimeError as exc:
            raise RuntimeError(f"{exc} from {start_day}") from None
        if not dry_run:
            sync_run_dir.mkdir(parents=True, exist_ok=True)
            sync_run_path.write_text(
                json.dumps(
                    {
                        "expected_retained_sources": expected_retained_sources,
                        "selected": [
                            {"name": name, "bytes": advertised_size}
                            for name, advertised_size in selected
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            volume.commit()
    if dry_run:
        return [
            {
                "name": name,
                "status": "planned",
                "bytes": advertised_size,
                "sha256": "",
            }
            for name, advertised_size in selected
        ]

    def sync_file(name: str, advertised_size: int) -> dict[str, int | str]:
        path = os.path.join(REMOTE_DATA_PATH, name)
        if os.path.exists(path) and os.path.getsize(path) == advertised_size:
            status = "exists"
        else:
            if os.path.exists(path):
                raise RuntimeError(
                    f"existing source {name} has {os.path.getsize(path)} bytes; "
                    f"upstream advertises {advertised_size}"
                )

            tmp_path = f"{path}.tmp"
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            url = f"{BASE_URL}/{name}"
            print(f"downloading {url}", flush=True)
            for attempt in range(10):
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 chess-engine-4-data-sync"},
                )
                try:
                    with (
                        urllib.request.urlopen(request) as response,
                        open(tmp_path, "wb") as handle,
                    ):
                        shutil.copyfileobj(response, handle, length=1024 * 1024)
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code != 429 or attempt == 9:
                        raise RuntimeError(f"failed to download {url}: HTTP {exc.code}") from None
                    retry_after = int(exc.headers.get("Retry-After", "60"))
                    time.sleep(max(retry_after, 60))
            actual_size = os.path.getsize(tmp_path)
            if actual_size != advertised_size:
                raise RuntimeError(
                    f"downloaded {name} has {actual_size} bytes; expected {advertised_size}"
                )
            os.replace(tmp_path, path)
            status = "downloaded"

        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        manifest_dir = Path(REMOTE_DATA_PATH) / "source-manifests"
        manifest_dir.mkdir(exist_ok=True)
        manifest_path = manifest_dir / f"{Path(name).stem}.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "source_name": name,
                    "source_url": f"{BASE_URL}/{name}",
                    "bytes": advertised_size,
                    "sha256": digest.hexdigest(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return {
            "name": name,
            "status": status,
            "bytes": advertised_size,
            "sha256": digest.hexdigest(),
        }

    if download_concurrency != 1:
        raise RuntimeError("only serial acquisition is supported")
    selected_names = {name for name, _ in selected}
    validate_sync_run_inventory(current_sources, retained_sources, selected_names)
    results = []
    for name, advertised_size in selected:
        current_sources = {
            entry.path
            for entry in volume.listdir("/")
            if entry.type == 1 and entry.path.endswith(".tar")
        }
        validate_sync_run_inventory(current_sources, retained_sources, selected_names)
        result = sync_file(name, advertised_size)
        results.append(result)
        print(
            f"{result['status']} {result['bytes']} {result['name']} sha256={result['sha256']}",
            flush=True,
        )
        volume.commit()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-day", default="20240401")
    parser.add_argument("--file-count", type=int, default=8)
    parser.add_argument("--minimum-source-mib", type=int, default=DEFAULT_MINIMUM_SOURCE_MIB)
    parser.add_argument("--download-concurrency", type=int, default=1)
    parser.add_argument(
        "--sync-run-id",
        help="resume the exact selection in an existing committed sync-run manifest",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.download_concurrency != 1:
        parser.error("--download-concurrency must be 1")

    retained_source_names = sorted(
        entry.path
        for entry in volume.listdir("/")
        if entry.type == 1 and entry.path.endswith(".tar")
    )

    rows: list[dict[str, int | str]] = []
    sync_run_id = args.sync_run_id or uuid.uuid4().hex
    with app.run():
        rows = sync_files.remote(
            args.start_day,
            args.file_count,
            args.minimum_source_mib * 2**20,
            args.dry_run,
            args.download_concurrency,
            retained_source_names,
            sync_run_id,
            args.sync_run_id is not None,
        )
    for row in rows:
        print(f"{row['status']} {row['bytes']} {row['name']} sha256={row['sha256']}")
    final_source_names = {
        entry.path
        for entry in volume.listdir("/")
        if entry.type == 1 and entry.path.endswith(".tar")
    }
    expected_source_names = set(retained_source_names) | {
        str(row["name"]) for row in rows if row["status"] != "planned"
    }
    if not args.dry_run and final_source_names != expected_source_names:
        raise RuntimeError(
            "postflight source inventory differs from the exact selected set: "
            f"expected={sorted(expected_source_names)} actual={sorted(final_source_names)}"
        )
    print(f"sync_complete files={len(rows)} exact_inventory=true")


if __name__ == "__main__":
    main()
