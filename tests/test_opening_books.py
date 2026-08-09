from pathlib import Path

import pytest

from chess_engine_4.opening_books import build_uho_sample


def test_build_uho_sample_is_deterministic_and_paired_format_compatible(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.epd"
    source.write_text("".join(f"fen-{index}\n" for index in range(20)))

    first = build_uho_sample(source, tmp_path / "first", sample_size=8, seed=7)
    second = build_uho_sample(source, tmp_path / "second", sample_size=8, seed=7)

    assert first["epd_sha256"] == second["epd_sha256"]
    assert first["pgn_sha256"] == second["pgn_sha256"]
    assert first["selected_indices_sha256"] == second["selected_indices_sha256"]
    assert (tmp_path / "first" / "UHO_Lichess_4852_v1-random-8.epd").read_text().count(
        "\n"
    ) == 8
    assert (tmp_path / "first" / "UHO_Lichess_4852_v1-random-8.pgn").read_text().count(
        '[FEN "'
    ) == 8


def test_build_uho_sample_rejects_insufficient_source(tmp_path: Path) -> None:
    source = tmp_path / "source.epd"
    source.write_text("a\nb\n")

    with pytest.raises(ValueError, match="only 2 rows"):
        build_uho_sample(source, tmp_path / "output", sample_size=3, seed=1)
