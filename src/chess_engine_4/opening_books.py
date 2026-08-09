"""Prepare reproducible opening-book samples for engine evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def prepare_uho_book() -> None:
    parser = argparse.ArgumentParser(
        description="Build matched EPD/PGN samples from UHO_Lichess_4852_v1."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sample-size", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    manifest = build_uho_sample(args.source, args.output_dir, args.sample_size, args.seed)
    print(json.dumps(manifest, indent=2))


def build_uho_sample(
    source: Path, output_dir: Path, sample_size: int, seed: int
) -> dict[str, int | str]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    rng = random.Random(seed)
    sample: list[tuple[int, str]] = []
    source_rows = 0
    with source.open() as source_file:
        for source_rows, raw_line in enumerate(source_file, start=1):
            line = raw_line.rstrip("\r\n")
            if source_rows <= sample_size:
                sample.append((source_rows - 1, line))
            else:
                replacement = rng.randrange(source_rows)
                if replacement < sample_size:
                    sample[replacement] = (source_rows - 1, line)
    if source_rows < sample_size:
        raise ValueError(f"source has only {source_rows} rows, need {sample_size}")

    sample.sort()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"UHO_Lichess_4852_v1-random-{sample_size}"
    epd_path = output_dir / f"{stem}.epd"
    pgn_path = output_dir / f"{stem}.pgn"
    epd_path.write_text("".join(f"{line}\n" for _, line in sample))
    pgn_path.write_text(
        "".join(
            f'[Event "UHO_Lichess_4852_v1 random sample"]\n'
            f'[Site "official-stockfish/books"]\n'
            f'[Round "{round_number}"]\n'
            f'[FEN "{line}"]\n'
            f'[SetUp "1"]\n\n*\n\n'
            for round_number, (_, line) in enumerate(sample, start=1)
        )
    )
    selected_indices = "".join(f"{index}\n" for index, _ in sample).encode()
    manifest: dict[str, int | str] = {
        "source": str(source),
        "source_rows": source_rows,
        "source_sha256": _sha256(source),
        "sample_size": sample_size,
        "sample_seed": seed,
        "selected_indices_sha256": hashlib.sha256(selected_indices).hexdigest(),
        "epd": str(epd_path),
        "epd_sha256": _sha256(epd_path),
        "pgn": str(pgn_path),
        "pgn_sha256": _sha256(pgn_path),
    }
    (output_dir / f"{stem}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
