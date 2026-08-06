#!/usr/bin/env python3
"""Clone or fast-forward the kernel development reference repositories."""

from __future__ import annotations

import argparse
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

REFERENCE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = REFERENCE_DIR / "repos.toml"
REPOS_DIR = REFERENCE_DIR / "repos"


@dataclass(frozen=True, slots=True)
class Repository:
    name: str
    url: str
    purpose: str


def main() -> None:
    repositories = _load_manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repositories",
        nargs="*",
        choices=[repository.name for repository in repositories],
        help="Repositories to sync; omit to sync all references.",
    )
    parser.add_argument("--list", action="store_true", help="List references without cloning.")
    args = parser.parse_args()

    selected = (
        [repository for repository in repositories if repository.name in args.repositories]
        if args.repositories
        else repositories
    )
    if args.list:
        for repository in selected:
            print(f"{repository.name}: {repository.purpose}")
        return

    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    for repository in selected:
        _sync(repository)


def _load_manifest() -> list[Repository]:
    with MANIFEST_PATH.open("rb") as file:
        manifest = tomllib.load(file)
    return [Repository(**entry) for entry in manifest["repository"]]


def _sync(repository: Repository) -> None:
    destination = REPOS_DIR / repository.name
    if not destination.exists():
        print(f"cloning {repository.name}", flush=True)
        _run(
            "git",
            "clone",
            "--depth=1",
            "--filter=blob:none",
            "--single-branch",
            repository.url,
            str(destination),
        )
        return

    if not (destination / ".git").is_dir():
        raise RuntimeError(f"{destination} exists but is not a Git repository")
    origin = _run(
        "git",
        "-C",
        str(destination),
        "remote",
        "get-url",
        "origin",
        capture_output=True,
    ).stdout.strip()
    if origin != repository.url:
        raise RuntimeError(f"{destination} has origin {origin!r}, expected {repository.url!r}")
    print(f"updating {repository.name}", flush=True)
    _run("git", "-C", str(destination), "pull", "--ff-only")


def _run(*command: str, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture_output,
    )


if __name__ == "__main__":
    main()
