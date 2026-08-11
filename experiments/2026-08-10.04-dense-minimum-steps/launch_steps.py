"""Launch matched B=32d/B=16d minimum-step ladders."""

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
CONFIG = ROOT / "configs/dense.py"
THROUGHPUT = ROOT / "experiments/throughput-dense.toml"
B16_LR_MULTIPLIER = {128: 0.85, 512: 1.00, 768: 1.15, 1024: 1.30}
PAIRED_HORIZONS = {
    128: (1_000, 2_000, 4_000),
    512: (1_000, 2_000, 4_000),
    768: (8_000, 16_000),
    1024: (4_000,),
}
B32_ONLY_HORIZONS = {
    128: (8_000,),
    512: (8_000, 16_000),
    768: (),
    1024: (8_000, 16_000),
}


@dataclass(frozen=True, slots=True)
class Trial:
    width: int
    equivalent_b32_steps: int
    batch_label: str
    training_ratio: float
    batch_size: int
    steps: int
    lr: float
    estimated_cost: float

    @property
    def name(self) -> str:
        return (
            f"dense-minsteps-horizon-d{self.width}-s{self.equivalent_b32_steps}"
            f"-{self.batch_label}"
        )

    def command(self, *, dry_run: bool) -> list[str]:
        command = [
            "uv",
            "run",
            "train-modal",
            "--config",
            str(CONFIG),
            "--d-model",
            str(self.width),
            "--training-ratio",
            f"{self.training_ratio:.12g}",
            "--batch-size",
            str(self.batch_size),
            "--steps",
            str(self.steps),
            "--lr",
            f"{self.lr:.12g}",
            "--wandb-name",
            self.name,
        ]
        if dry_run:
            command.append("--dry-run")
        return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("width", type=int, choices=B16_LR_MULTIPLIER)
    parser.add_argument("--launch", action="store_true", help="Launch Modal jobs.")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Launch only the 16,000-step B=32d validation point.",
    )
    args = parser.parse_args()

    _load_main_environment()
    trials = _trials(args.width)
    if args.validation_only:
        trials = [
            trial
            for trial in trials
            if trial.batch_label == "b32" and trial.equivalent_b32_steps == 16_000
        ]
        if not trials:
            parser.error("this width has no validation-only trial")
    total_cost = sum(trial.estimated_cost for trial in trials)
    print(
        f"stage=steps-d{args.width} trials={len(trials)} "
        f"conservative_estimated_cost=${total_cost:.3f}"
    )
    for trial in trials:
        print(
            f"trial={trial.name} ratio={trial.training_ratio:.8g} "
            f"batch={trial.batch_size} steps={trial.steps} lr={trial.lr:.8g} "
            f"estimated_cost=${trial.estimated_cost:.3f}"
        )

    dry_run = not args.launch
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(trials)) as executor:
        results = list(
            executor.map(
                lambda trial: subprocess.run(
                    trial.command(dry_run=dry_run), cwd=ROOT, check=False
                ),
                trials,
            )
        )
    failures = [result.returncode for result in results if result.returncode != 0]
    if failures:
        raise SystemExit(f"{len(failures)} trial(s) failed: {failures}")


def _trials(width: int) -> list[Trial]:
    with THROUGHPUT.open("rb") as file:
        row = tomllib.load(file)["models"][f"d{width}"]
    steps_1x = int(row["steps_1x"])
    rate = hardware_dollars_per_second(str(row["gpu"]), int(row["cpu_cores"]))
    seconds_per_step = float(row["measured_wall_ms_per_step"]) / 1_000
    trials: list[Trial] = []
    for equivalent_steps in PAIRED_HORIZONS[width]:
        training_ratio = equivalent_steps / steps_1x
        config = load_training_config(
            CONFIG, d_model=width, training_ratio=training_ratio
        )
        batch32 = config.run.batch_size
        trials.append(
            Trial(
                width=width,
                equivalent_b32_steps=equivalent_steps,
                batch_label="b32",
                training_ratio=training_ratio,
                batch_size=batch32,
                steps=equivalent_steps,
                lr=config.optimizer.lr,
                estimated_cost=seconds_per_step * equivalent_steps * rate,
            )
        )
        steps16 = equivalent_steps * 2
        trials.append(
            Trial(
                width=width,
                equivalent_b32_steps=equivalent_steps,
                batch_label="b16",
                training_ratio=training_ratio,
                batch_size=batch32 // 2,
                steps=steps16,
                lr=config.optimizer.lr * B16_LR_MULTIPLIER[width],
                estimated_cost=seconds_per_step * steps16 * rate,
            )
        )
    for equivalent_steps in B32_ONLY_HORIZONS[width]:
        training_ratio = equivalent_steps / steps_1x
        config = load_training_config(
            CONFIG, d_model=width, training_ratio=training_ratio
        )
        trials.append(
            Trial(
                width=width,
                equivalent_b32_steps=equivalent_steps,
                batch_label="b32",
                training_ratio=training_ratio,
                batch_size=config.run.batch_size,
                steps=equivalent_steps,
                lr=config.optimizer.lr,
                estimated_cost=seconds_per_step * equivalent_steps * rate,
            )
        )
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
