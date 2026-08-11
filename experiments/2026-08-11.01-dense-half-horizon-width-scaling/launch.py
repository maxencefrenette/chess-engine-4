"""Launch the minimum-safe dense width-scaling ladder."""

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
TRAINING_RATIO = 0.055
WIDTHS = (64, 128, 256, 512, 768, 1024)
RETRY_LRS = {
    512: (0.85, 0.000493),
    768: (1.00, 0.00033),
    1024: (1.15, 0.000253),
}


@dataclass(frozen=True, slots=True)
class Trial:
    width: int
    estimated_cost: float
    multiplier: float | None = None
    lr: float | None = None

    @property
    def name(self) -> str:
        if self.multiplier is not None:
            return (
                f"dense-half-width-d{self.width}-r0p055-"
                f"lr{round(100 * self.multiplier):03d}"
            )
        return f"dense-half-width-d{self.width}-r0p055"

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
        if self.lr is not None:
            command.extend(("--lr", f"{self.lr:.12g}"))
        if dry_run:
            command.append("--dry-run")
        return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true", help="Launch Modal jobs.")
    parser.add_argument(
        "--retry-lr", action="store_true", help="Retry spiked widths at lower LRs."
    )
    args = parser.parse_args()

    _load_main_environment()
    trials = _trials(retry_lr=args.retry_lr)
    print(
        f"trials={len(trials)} ratio={TRAINING_RATIO:g} "
        f"conservative_estimated_cost=${sum(t.estimated_cost for t in trials):.3f}"
    )
    for trial in trials:
        config = load_training_config(CONFIG, d_model=trial.width, training_ratio=TRAINING_RATIO)
        lr = config.optimizer.lr if trial.lr is None else trial.lr
        print(
            f"trial={trial.name} batch={config.run.batch_size} steps={config.run.steps} "
            f"lr={lr:.8g} estimated_cost=${trial.estimated_cost:.3f}"
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


def _trials(*, retry_lr: bool) -> list[Trial]:
    with THROUGHPUT.open("rb") as file:
        models = tomllib.load(file)["models"]
    trials = []
    widths = tuple(RETRY_LRS) if retry_lr else WIDTHS
    for width in widths:
        row = models[f"d{width}"]
        config = load_training_config(CONFIG, d_model=width, training_ratio=TRAINING_RATIO)
        rate = hardware_dollars_per_second(str(row["gpu"]), int(row["cpu_cores"]))
        cost = float(row["measured_wall_ms_per_step"]) / 1_000 * config.run.steps * rate
        retry = RETRY_LRS.get(width) if retry_lr else None
        trials.append(
            Trial(
                width=width,
                estimated_cost=cost,
                multiplier=None if retry is None else retry[0],
                lr=None if retry is None else retry[1],
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
