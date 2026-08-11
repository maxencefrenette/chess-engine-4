from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "sync_modal_training_data.py"
SPEC = importlib.util.spec_from_file_location("sync_modal_training_data", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


def test_select_source_candidates_returns_exact_requested_prefix() -> None:
    candidates = [(f"source-{index}.tar", 100) for index in range(5)]

    assert sync.select_source_candidates(candidates, 3) == candidates[:3]


def test_select_source_candidates_refuses_short_candidate_list() -> None:
    candidates = [(f"source-{index}.tar", 100) for index in range(2)]

    with pytest.raises(RuntimeError, match="requested 3 files, but only 2 candidates"):
        sync.select_source_candidates(candidates, 3)


def test_unexpected_complete_sources_allows_only_initial_and_selected() -> None:
    retained = {"retained.tar"}
    selected = {"selected-a.tar", "selected-b.tar"}

    assert sync.unexpected_complete_sources(
        retained | selected, retained, selected
    ) == set()
    assert sync.unexpected_complete_sources(
        retained | selected | {"residual.tar"}, retained, selected
    ) == {"residual.tar"}


def test_validate_sync_run_inventory_accepts_resumed_selected_sources() -> None:
    retained = {"retained.tar"}
    selected = {"selected-a.tar", "selected-b.tar"}

    sync.validate_sync_run_inventory(
        retained | {"selected-a.tar"}, retained, selected
    )
    with pytest.raises(RuntimeError, match="unexpected complete sources"):
        sync.validate_sync_run_inventory(
            retained | selected | {"outside-run.tar"}, retained, selected
        )


def test_explicit_resume_uses_manifest_retained_source_baseline() -> None:
    assert sync.resolve_retained_sources_for_run(
        [],
        ["selected-source-already-complete.tar"],
        resume_existing_run=True,
    ) == set()


def test_implicit_retry_requires_matching_retained_source_baseline() -> None:
    with pytest.raises(RuntimeError, match="retained-source inventory"):
        sync.resolve_retained_sources_for_run(
            [],
            ["outside-run.tar"],
            resume_existing_run=False,
        )
