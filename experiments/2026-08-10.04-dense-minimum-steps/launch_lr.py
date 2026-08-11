"""Launch the cost-gated B=16d learning-rate calibration ladder."""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from chess_engine_4.hardware import hardware_dollars_per_second

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/dense.py"
THROUGHPUT = ROOT / "experiments/throughput-dense.toml"
EQUIVALENT_B32_STEPS = 8_000
EQUIVALENT_B32_STEPS_BY_STAGE = {"d1280": 12_000}
STAGES = {
    "initial": {128: (0.55, 0.70, 0.85), 512: (0.55, 0.70, 0.85), 1024: (0.70,)},
    "edge1": {128: (1.00,), 512: (1.00,), 1024: (0.85, 1.00)},
    "edge2": {128: (1.15,), 512: (1.15,), 1024: (1.15, 1.30)},
    "edge3": {1024: (1.50, 1.70)},
    "d1280": {1280: (1.15, 1.30)},
}
LR_PARAMETER_COEFFICIENT = 31.75
LR_PARAMETER_EXPONENT = -0.74
LR_TRAINING_RATIO_EXPONENT = -0.63


@dataclass(frozen=True, slots=True)
class Trial:
    width: int
    training_ratio: float
    batch_size: int
    steps: int
    lr: float
    multiplier: float
    estimated_cost: float

    @property
    def name(self) -> str:
        suffix = round(100 * self.multiplier)
        return f"dense-minsteps-lr16d-d{self.width}-m{suffix:03d}"

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
    parser.add_argument("--stage", choices=STAGES, default="initial")
    parser.add_argument("--launch", action="store_true", help="Launch Modal jobs.")
    args = parser.parse_args()

    _load_main_environment()
    equivalent_b32_steps = EQUIVALENT_B32_STEPS_BY_STAGE.get(
        args.stage, EQUIVALENT_B32_STEPS
    )
    trials = _trials(STAGES[args.stage], equivalent_b32_steps)
    total_cost = sum(trial.estimated_cost for trial in trials)
    print(
        f"stage=lr-{args.stage} trials={len(trials)} "
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


def _trials(
    multipliers_by_width: dict[int, tuple[float, ...]], equivalent_b32_steps: int
) -> list[Trial]:
    with THROUGHPUT.open("rb") as file:
        models = tomllib.load(file)["models"]
    trials: list[Trial] = []
    for width, multipliers in multipliers_by_width.items():
        row = models[f"d{width}"]
        training_ratio = equivalent_b32_steps / int(row["steps_1x"])
        base_lr = _round_significant(
            LR_PARAMETER_COEFFICIENT
            * int(row["params"]) ** LR_PARAMETER_EXPONENT
            * training_ratio**LR_TRAINING_RATIO_EXPONENT,
            digits=2,
        )
        batch_size = int(row["batch_size"]) // 2
        steps = equivalent_b32_steps * 2
        rate = hardware_dollars_per_second(
            str(row["gpu"]), int(row["cpu_cores"])
        )
        estimated_cost = float(row["measured_wall_ms_per_step"]) / 1_000 * steps * rate
        trials.extend(
            Trial(
                width=width,
                training_ratio=training_ratio,
                batch_size=batch_size,
                steps=steps,
                lr=base_lr * multiplier,
                multiplier=multiplier,
                estimated_cost=estimated_cost,
            )
            for multiplier in multipliers
        )
    return trials


def _round_significant(value: float, *, digits: int) -> float:
    places = digits - 1 - math.floor(math.log10(abs(value)))
    return round(value, places)


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
