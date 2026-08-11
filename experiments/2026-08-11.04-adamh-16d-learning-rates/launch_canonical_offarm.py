"""Rerun affected 0.1x scaling cells with the canonical dense 16d recipe."""

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
THROUGHPUT = ROOT / "experiments/throughput-dense-16d.toml"
TRAINING_RATIO = 0.1
WIDTHS = (256, 512)


@dataclass(frozen=True, slots=True)
class Trial:
    width: int
    batch_size: int
    steps: int
    lr: float
    estimated_cost: float

    @property
    def name(self) -> str:
        lr_name = f"{self.lr:.8f}".rstrip("0").replace(".", "p")
        return f"adamh-16d-canonical-d{self.width}-r0p1-lr{lr_name}"

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
            str(TRAINING_RATIO),
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
        f"stage=canonical-offarm trials={len(trials)} ratio={TRAINING_RATIO:g} "
        f"conservative_estimated_cost=${sum(t.estimated_cost for t in trials):.3f}"
    )
    for trial in trials:
        print(
            f"trial={trial.name} batch={trial.batch_size} steps={trial.steps} "
            f"lr={trial.lr:.8g} estimated_cost=${trial.estimated_cost:.3f}"
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(trials)) as executor:
        results = list(
            executor.map(
                lambda trial: subprocess.run(
                    trial.command(dry_run=not args.launch), cwd=ROOT, check=False
                ),
                trials,
            )
        )
    failures = [result.returncode for result in results if result.returncode != 0]
    if failures:
        raise SystemExit(f"{len(failures)} trial(s) failed: {failures}")


def _trials() -> list[Trial]:
    with THROUGHPUT.open("rb") as file:
        models = tomllib.load(file)["models"]
    trials: list[Trial] = []
    for width in WIDTHS:
        config = load_training_config(
            CONFIG, d_model=width, training_ratio=TRAINING_RATIO
        )
        if config.run.batch_size != 16 * width:
            raise RuntimeError(f"d{width} did not select exact batch 16d")
        row = models[f"d{width}"]
        if int(row["batch_size"]) != config.run.batch_size:
            raise RuntimeError(f"d{width} lacks an exact-16d throughput row")
        rate = hardware_dollars_per_second(str(row["gpu"]), int(row["cpu_cores"]))
        cost = float(row["measured_wall_ms_per_step"]) / 1_000 * config.run.steps * rate
        trials.append(
            Trial(
                width=width,
                batch_size=config.run.batch_size,
                steps=config.run.steps,
                lr=config.optimizer.lr,
                estimated_cost=cost,
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
