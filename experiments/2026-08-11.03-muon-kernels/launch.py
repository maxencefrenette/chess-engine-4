"""Launch batched-Muon equivalence and light LR-tuning runs."""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from chess_engine_4.hardware import hardware_dollars_per_second
from chess_engine_4.training.config import load_training_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/2026-08-11.02-muon/config.py"
TRAINING_RATIO = 0.055
TRIALS = ((256, 0.5), (256, 0.75), (512, 0.5), (512, 0.75))
OLD_MUON_RUNTIME = {256: 43.018561091, 512: 114.267320601}
PROJECTED_SPEEDUP = {256: 7.425 / 5.125, 512: 11.406 / 7.858}


@dataclass(frozen=True, slots=True)
class Trial:
    width: int
    lr_multiplier: float
    estimated_cost: float

    @property
    def name(self) -> str:
        multiplier = round(100 * self.lr_multiplier)
        return f"muon-batched-d{self.width}-r0p055-lr{multiplier:03d}"

    def command(self, *, dry_run: bool) -> list[str]:
        config = load_training_config(CONFIG, d_model=self.width)
        command = [
            "uv",
            "run",
            "train-modal",
            "--config",
            str(CONFIG),
            "--d-model",
            str(self.width),
            "--training-ratio",
            str(TRAINING_RATIO),
            "--lr",
            f"{config.optimizer.lr * self.lr_multiplier:.12g}",
            "--wandb-name",
            self.name,
        ]
        if dry_run:
            command.append("--dry-run")
        return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true", help="Launch Modal jobs.")
    args = parser.parse_args()

    _load_main_environment()
    trials = _trials()
    print(
        f"trials={len(trials)} ratio={TRAINING_RATIO:g} "
        f"projected_cost=${sum(trial.estimated_cost for trial in trials):.3f}"
    )
    for trial in trials:
        config = load_training_config(CONFIG, d_model=trial.width)
        print(
            f"trial={trial.name} optimizer=batched_muon batch={config.run.batch_size} "
            f"steps={config.run.steps} lr={config.optimizer.lr * trial.lr_multiplier:.8g} "
            f"projected_cost=${trial.estimated_cost:.3f}"
        )

    if not args.launch:
        for trial in trials:
            subprocess.run(trial.command(dry_run=True), cwd=ROOT, check=True)
        return
    results = [
        subprocess.run(trial.command(dry_run=False), cwd=ROOT) for trial in trials
    ]
    failures = [result.returncode for result in results if result.returncode != 0]
    if failures:
        raise SystemExit(f"{len(failures)} trial(s) failed: {failures}")


def _trials() -> list[Trial]:
    trials = []
    for width, multiplier in TRIALS:
        config = load_training_config(CONFIG, d_model=width)
        rate = hardware_dollars_per_second(config.infra.gpu, config.infra.cpu_cores)
        runtime = OLD_MUON_RUNTIME[width] / PROJECTED_SPEEDUP[width]
        trials.append(Trial(width, multiplier, runtime * rate))
    return trials


def _load_main_environment() -> None:
    common_dir = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"], cwd=ROOT, text=True
        ).strip()
    ).resolve()
    load_dotenv(common_dir.parent / ".env", override=False)
    if "WANDB_PROJECT" not in os.environ:
        raise RuntimeError("WANDB_PROJECT is not configured in the main repository .env")


if __name__ == "__main__":
    main()
