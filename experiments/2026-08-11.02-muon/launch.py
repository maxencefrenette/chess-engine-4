"""Launch a cheap Muon LR screen or matched width comparison."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from chess_engine_4.hardware import hardware_dollars_per_second
from chess_engine_4.training.config import load_training_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("config.py")
THROUGHPUT = ROOT / "experiments/throughput-dense.toml"
TRAINING_RATIO = 0.055
SCREEN_MULTIPLIERS = (0.5, 1.0, 2.0)
WIDTH_LR_MULTIPLIER = 0.5
WIDTHS = (128, 256, 512)


@dataclass(frozen=True, slots=True)
class Trial:
    width: int
    lr_multiplier: float
    estimated_adam_cost: float

    @property
    def name(self) -> str:
        multiplier = round(100 * self.lr_multiplier)
        return f"muon-d{self.width}-r0p055-lr{multiplier:03d}"

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
    parser.add_argument(
        "--widths",
        action="store_true",
        help="Run the matched d128/d256/d512 comparison at the selected 1x LR.",
    )
    args = parser.parse_args()

    _load_main_environment()
    trials = _trials(widths=args.widths)
    total_adam_cost = sum(trial.estimated_adam_cost for trial in trials)
    print(
        f"phase={'widths' if args.widths else 'screen'} trials={len(trials)} "
        f"ratio={TRAINING_RATIO:g} adam_cost_reference=${total_adam_cost:.3f}"
    )
    for trial in trials:
        config = load_training_config(CONFIG, d_model=trial.width)
        print(
            f"trial={trial.name} optimizer={config.optimizer.algorithm} "
            f"batch={config.run.batch_size} steps={config.run.steps} "
            f"lr={config.optimizer.lr * trial.lr_multiplier:.8g} "
            f"adam_cost_reference=${trial.estimated_adam_cost:.3f}"
        )

    if not args.launch:
        for trial in trials:
            subprocess.run(trial.command(dry_run=True), cwd=ROOT, check=True)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(trials)) as executor:
        results = list(
            executor.map(
                lambda trial: subprocess.run(trial.command(dry_run=False), cwd=ROOT),
                trials,
            )
        )
    failures = [result.returncode for result in results if result.returncode != 0]
    if failures:
        raise SystemExit(f"{len(failures)} trial(s) failed: {failures}")


def _trials(*, widths: bool) -> list[Trial]:
    with THROUGHPUT.open("rb") as file:
        models = tomllib.load(file)["models"]
    pairs = ((width, WIDTH_LR_MULTIPLIER) for width in WIDTHS) if widths else (
        (64, multiplier) for multiplier in SCREEN_MULTIPLIERS
    )
    trials = []
    for width, multiplier in pairs:
        config = load_training_config(CONFIG, d_model=width)
        row = models[f"d{width}"]
        rate = hardware_dollars_per_second(str(row["gpu"]), int(row["cpu_cores"]))
        cost = float(row["measured_wall_ms_per_step"]) / 1_000 * config.run.steps * rate
        trials.append(Trial(width, multiplier, cost))
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
