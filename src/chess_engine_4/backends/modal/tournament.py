"""Run the shared tournament analysis through Modal jobs."""

import argparse
import json
from pathlib import Path

from chess_engine_4.backends.modal.eval import app, selfplay_eval_function
from chess_engine_4.backends.modal.paths import lc0_path_for_gpu
from chess_engine_4.evaluation.tournament import (
    _load_completed_matches,
    _write_report,
    build_match_payloads,
    build_report,
    load_tournament_config,
    parse_match_result,
    select_pairings,
)


def eval_tournament_modal() -> None:
    parser = argparse.ArgumentParser(description="Run an adaptive lc0 tournament on Modal.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tournament, engines = load_tournament_config(args.config)
    matches = _load_completed_matches(args.output, tournament, engines)
    completed_waves = max((match.wave for match in matches), default=-1) + 1
    if completed_waves < tournament.waves:
        function = selfplay_eval_function(
            tournament.gpu,
            max_containers=tournament.max_concurrency,
        )
        with app.run():
            for wave in range(completed_waves, tournament.waves):
                pairings = select_pairings(
                    engines, matches, wave, tournament.games_per_matchup
                )
                payloads = build_match_payloads(
                    tournament, pairings, wave, lc0_path=str(lc0_path_for_gpu(tournament.gpu))
                )
                raw_results = list(function.map(payloads))
                matches.extend(
                    parse_match_result(payload, result)
                    for payload, result in zip(payloads, raw_results, strict=True)
                )
                report = build_report(tournament, engines, matches)
                _write_report(args.output, report)
                print(
                    f"completed wave {wave + 1}/{tournament.waves}: "
                    + ", ".join(f"{left.name} vs {right.name}" for left, right in pairings)
                )

    report = build_report(tournament, engines, matches)
    _write_report(args.output, report)
    print(json.dumps(report, indent=2))
