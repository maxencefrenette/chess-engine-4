#!/usr/bin/env python3
"""Download the project research-paper references."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REFERENCE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = REFERENCE_DIR / "papers.toml"
PAPERS_DIR = REFERENCE_DIR / "papers"


@dataclass(frozen=True, slots=True)
class Paper:
    slug: str
    title: str
    authors: str
    citation: str
    url: str
    filename: str
    relevance: str


def main() -> None:
    papers = _load_manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "papers",
        nargs="*",
        choices=[paper.slug for paper in papers],
        help="Papers to download; omit to download all references.",
    )
    parser.add_argument("--list", action="store_true", help="List papers without downloading.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Replace PDFs that have already been downloaded.",
    )
    args = parser.parse_args()

    selected = [paper for paper in papers if paper.slug in args.papers] if args.papers else papers
    if args.list:
        for paper in selected:
            print(f"{paper.slug}: {paper.title} ({paper.citation})")
        return

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    for paper in selected:
        _download(paper, refresh=args.refresh)


def _load_manifest() -> list[Paper]:
    with MANIFEST_PATH.open("rb") as file:
        manifest = tomllib.load(file)
    return [Paper(**entry) for entry in manifest["paper"]]


def _download(paper: Paper, *, refresh: bool) -> None:
    destination = PAPERS_DIR / paper.filename
    if destination.exists() and not refresh:
        print(f"cached {paper.slug}")
        return

    print(f"downloading {paper.slug}", flush=True)
    request = urllib.request.Request(
        paper.url,
        headers={"User-Agent": "chess-engine-4 reference sync"},
    )
    temporary_path: Path | None = None
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            tempfile.NamedTemporaryFile(dir=PAPERS_DIR, delete=False) as temporary,
        ):
            temporary_path = Path(temporary.name)
            shutil.copyfileobj(response, temporary)
        with temporary_path.open("rb") as downloaded:
            if downloaded.read(5) != b"%PDF-":
                raise RuntimeError(f"download for {paper.slug} is not a PDF")
        temporary_path.replace(destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
