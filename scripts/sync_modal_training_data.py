"""Sync LCZero training tar files directly into the Modal training-data Volume."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import modal

APP_NAME = "chess-engine-4-data-sync"
DATA_VOLUME_NAME = "chess-engine-4-training-data"
ARTIFACT_VOLUME_NAME = "chess-engine-4-artifacts"
REMOTE_DATA_PATH = "/data/training_data"
BASE_URL = "https://data.lczero.org/files/training_data/test80"
DEFAULT_OPERATIONAL_CEILING_GIB = 900
DEFAULT_MINIMUM_SOURCE_MIB = 100

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME)


@app.function(
    image=modal.Image.debian_slim(python_version="3.14"),
    volumes={REMOTE_DATA_PATH: volume},
    timeout=24 * 60 * 60,
)
def sync_files(
    start_day: str,
    file_count: int,
    minimum_source_bytes: int,
    available_workspace_bytes: int,
    dry_run: bool,
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
    retained_sources = {
        entry.path
        for entry in volume.listdir("/")
        if entry.type == 1 and entry.path.endswith(".tar")
    }
    candidates = sorted(
        (name, advertised_sizes[name])
        for href in parser.hrefs
        if (match := filename_pattern.fullmatch(name := href.rsplit("/", 1)[-1]))
        and match.group(1) >= start_day
        and name.removesuffix(".tar") not in converted
        and name not in retained_sources
        and advertised_sizes.get(name, 0) >= minimum_source_bytes
    )
    selected: list[tuple[str, int]] = []
    reserved_bytes = 0
    for name, advertised_size in candidates:
        # Reserve one source-sized output allocation as a deliberately conservative
        # bound for the later Parquet conversion.
        projected_bytes = advertised_size * 2
        if reserved_bytes + projected_bytes > available_workspace_bytes:
            break
        selected.append((name, advertised_size))
        reserved_bytes += projected_bytes
        if len(selected) == file_count:
            break
    if len(selected) < file_count:
        raise RuntimeError(
            f"requested {file_count} safe files from {start_day}, but only {len(selected)} fit "
            "the source-size and workspace-capacity constraints"
        )
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

    results = []
    for index, (name, advertised_size) in enumerate(selected, start=1):
        result = sync_file(name, advertised_size)
        results.append(result)
        print(
            f"{result['status']} {result['bytes']} {result['name']} sha256={result['sha256']}",
            flush=True,
        )
        if index % 24 == 0 or index == len(selected):
            volume.commit()
        if result["status"] == "downloaded" and index != len(selected):
            time.sleep(5)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-day", default="20240401")
    parser.add_argument("--file-count", type=int, default=8)
    parser.add_argument(
        "--operational-ceiling-gib", type=int, default=DEFAULT_OPERATIONAL_CEILING_GIB
    )
    parser.add_argument("--minimum-source-mib", type=int, default=DEFAULT_MINIMUM_SOURCE_MIB)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    training_bytes = _volume_bytes(volume)
    artifact_bytes = _volume_bytes(artifact_volume)
    converted_names = {
        Path(entry.path).stem
        for entry in volume.listdir("/parquet")
        if entry.type == 1 and entry.path.endswith(".parquet")
    }
    unconverted_source_bytes = sum(
        entry.size
        for entry in volume.listdir("/")
        if entry.type == 1
        and entry.path.endswith(".tar")
        and Path(entry.path).stem not in converted_names
    )
    ceiling_bytes = args.operational_ceiling_gib * 2**30
    available_bytes = (
        ceiling_bytes - training_bytes - artifact_bytes - unconverted_source_bytes
    )
    if available_bytes <= 0:
        parser.error("combined Modal Volume usage is already at the operational ceiling")
    print(
        f"storage_preflight training_bytes={training_bytes} artifact_bytes={artifact_bytes} "
        f"combined_bytes={training_bytes + artifact_bytes} ceiling_bytes={ceiling_bytes} "
        f"unconverted_output_reserve_bytes={unconverted_source_bytes} "
        f"available_bytes={available_bytes}"
    )

    with app.run():
        rows = sync_files.remote(
            args.start_day,
            args.file_count,
            args.minimum_source_mib * 2**20,
            available_bytes,
            args.dry_run,
        )
    for row in rows:
        print(f"{row['status']} {row['bytes']} {row['name']} sha256={row['sha256']}")
    final_training_bytes = _volume_bytes(volume)
    final_artifact_bytes = _volume_bytes(artifact_volume)
    final_combined_bytes = final_training_bytes + final_artifact_bytes
    print(
        f"storage_postflight training_bytes={final_training_bytes} "
        f"artifact_bytes={final_artifact_bytes} combined_bytes={final_combined_bytes} "
        f"ceiling_bytes={ceiling_bytes} headroom_bytes={ceiling_bytes - final_combined_bytes}"
    )
    if final_combined_bytes > ceiling_bytes:
        raise RuntimeError("combined Modal Volume usage exceeded the operational ceiling")


def _volume_bytes(remote_volume: modal.Volume) -> int:
    return sum(
        entry.size for entry in remote_volume.listdir("/", recursive=True) if entry.type == 1
    )


if __name__ == "__main__":
    main()
