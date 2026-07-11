"""Sync LCZero training tar files directly into the Modal training-data Volume."""

from __future__ import annotations

import argparse

import modal

APP_NAME = "chess-engine-4-data-sync"
DATA_VOLUME_NAME = "chess-engine-4-training-data"
REMOTE_DATA_PATH = "/data/training_data"
BASE_URL = "https://data.lczero.org/files/training_data/test80"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=modal.Image.debian_slim(python_version="3.14"),
    volumes={REMOTE_DATA_PATH: volume},
    timeout=24 * 60 * 60,
)
def sync_files(start_day: str, file_count: int) -> list[tuple[str, str, int]]:
    import datetime
    import os
    import shutil
    import time
    import urllib.error
    import urllib.request

    os.makedirs(REMOTE_DATA_PATH, exist_ok=True)
    start = datetime.datetime.strptime(start_day, "%Y%m%d").date()
    names = [
        f"training-run1-test80-{start + datetime.timedelta(days=index // 24):%Y%m%d}-"
        f"{index % 24:02d}17.tar"
        for index in range(file_count)
    ]

    def sync_file(name: str) -> tuple[str, str, int]:
        path = os.path.join(REMOTE_DATA_PATH, name)
        if os.path.exists(path) and os.path.getsize(path) > 1024 * 1024:
            return name, "exists", os.path.getsize(path)

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
                with urllib.request.urlopen(request) as response, open(tmp_path, "wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt == 9:
                    raise RuntimeError(f"failed to download {url}: HTTP {exc.code}") from None
                retry_after = int(exc.headers.get("Retry-After", "60"))
                time.sleep(max(retry_after, 60))
        os.replace(tmp_path, path)
        return name, "downloaded", os.path.getsize(path)

    results = []
    for index, name in enumerate(names, start=1):
        result = sync_file(name)
        results.append(result)
        print(f"{result[1]} {result[2]} {result[0]}", flush=True)
        if index % 24 == 0 or index == len(names):
            volume.commit()
        if result[1] == "downloaded" and index != len(names):
            time.sleep(5)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-day", default="20240401")
    parser.add_argument("--file-count", type=int, default=24)
    args = parser.parse_args()

    with app.run():
        rows = sync_files.remote(args.start_day, args.file_count)
    for name, status, size in rows:
        print(f"{status} {size} {name}")


if __name__ == "__main__":
    main()
