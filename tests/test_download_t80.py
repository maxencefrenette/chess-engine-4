from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "download_t80.py"
SPEC = importlib.util.spec_from_file_location("download_t80", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


def test_parse_inventory_html_extracts_names_and_sizes() -> None:
    content = b"""
<a href="training-run1-test80-20240410-0017.tar">archive</a>  1676042240
<a href="unrelated.tar">unrelated</a>  123
<a href="training-run1-test80-20240410-0117.tar">archive</a>  1683322880
"""

    assert download.parse_inventory_html(content) == [
        ("training-run1-test80-20240410-0017.tar", 1_676_042_240),
        ("training-run1-test80-20240410-0117.tar", 1_683_322_880),
    ]


def test_read_inventory_rejects_invalid_archive_name(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.tsv"
    inventory.write_text("other.tar\t100\n")

    with pytest.raises(ValueError, match="invalid archive name"):
        download.read_inventory(inventory)


@pytest.mark.parametrize(
    ("offset", "status", "content_range", "expected"),
    [
        (0, 200, None, "wb"),
        (100, 200, None, "wb"),
        (100, 206, "bytes 100-199/200", "ab"),
    ],
)
def test_response_write_mode(
    offset: int,
    status: int,
    content_range: str | None,
    expected: str,
) -> None:
    assert download.response_write_mode(
        offset=offset,
        status=status,
        content_range=content_range,
    ) == expected


def test_response_write_mode_rejects_wrong_resume_offset() -> None:
    with pytest.raises(RuntimeError, match="Content-Range"):
        download.response_write_mode(
            offset=100,
            status=206,
            content_range="bytes 0-199/200",
        )
