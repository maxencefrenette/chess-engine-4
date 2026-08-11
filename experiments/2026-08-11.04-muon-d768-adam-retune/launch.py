"""Retune d256 AdamW and test lower-LR batched Muon at d768."""

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
ADAM_CONFIG = ROOT / "configs/dense.py"
MUON_CONFIG = ROOT / "experiments/2026-08-11.02-muon/config.py"
TRAINING_RATIO = 0.055
ADAM_RUNTIME_D256 = 23.842348
PROJECTED_MUON_RUNTIME_D768 = 172.0


@dataclass(frozen=True, slots=True)
class Trial:
    optimizer: str
    width: int
    lr_multiplier: float
    estimated_cost: float
    suffix: str = ""

    @property
    def config_path(self) -> Path:
        return ADAM_CONFIG if self.optimizer == "adamw" else MUON_CONFIG

    @property
    def name(self) -> str:
        multiplier = round(100 * self.lr_multiplier)
        return f"{self.optimizer}-retune-d{self.width}-r0p055-lr{multiplier:03d}{self.suffix}"

    def command(self, *, dry_run: bool) -> list[str]:
        config = load_training_config(
            self.config_path,
            d_model=self.width,
            training_ratio=TRAINING_RATIO,
        )
        command = [
            "uv",
            "run",
            "train-modal",
            "--config",
            str(self.config_path),
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
        "--retry-interrupted",
        action="store_true",
        help="Retry only the three arms interrupted during the initial launch.",
    )
    parser.add_argument(
        "--adam-lower",
        action="store_true",
        help="Extend the d256 AdamW sweep below the initial best boundary.",
    )
    parser.add_argument(
        "--muon-upper",
        action="store_true",
        help="Extend the d768 Muon sweep above the initial best boundary.",
    )
    parser.add_argument(
        "--muon-one",
        action="store_true",
        help="Test the canonical 1x LR after the d768 0.75x boundary improved.",
    )
    parser.add_argument(
        "--muon-midpoint",
        action="store_true",
        help="Test 0.875x between the stable 0.75x and spiking 1x d768 arms.",
    )
    args = parser.parse_args()

    _load_main_environment()
    trials = _trials(
        retry_interrupted=args.retry_interrupted,
        adam_lower=args.adam_lower,
        muon_upper=args.muon_upper,
        muon_one=args.muon_one,
        muon_midpoint=args.muon_midpoint,
    )
    print(
        f"trials={len(trials)} ratio={TRAINING_RATIO:g} "
        f"estimated_cost=${sum(trial.estimated_cost for trial in trials):.3f}"
    )
    for trial in trials:
        config = load_training_config(
            trial.config_path,
            d_model=trial.width,
            training_ratio=TRAINING_RATIO,
        )
        print(
            f"trial={trial.name} optimizer={trial.optimizer} "
            f"batch={config.run.batch_size} steps={config.run.steps} "
            f"lr={config.optimizer.lr * trial.lr_multiplier:.8g} "
            f"estimated_cost=${trial.estimated_cost:.3f}"
        )

    results = [
        subprocess.run(trial.command(dry_run=not args.launch), cwd=ROOT)
        for trial in trials
    ]
    failures = [result.returncode for result in results if result.returncode != 0]
    if failures:
        raise SystemExit(f"{len(failures)} trial(s) failed: {failures}")


def _trials(
    *,
    retry_interrupted: bool,
    adam_lower: bool,
    muon_upper: bool,
    muon_one: bool,
    muon_midpoint: bool,
) -> list[Trial]:
    adam = load_training_config(ADAM_CONFIG, d_model=256, training_ratio=TRAINING_RATIO)
    muon = load_training_config(MUON_CONFIG, d_model=768, training_ratio=TRAINING_RATIO)
    adam_rate = hardware_dollars_per_second(adam.infra.gpu, adam.infra.cpu_cores)
    muon_rate = hardware_dollars_per_second(muon.infra.gpu, muon.infra.cpu_cores)
    trials = [
        *(
            Trial("adamw", 256, multiplier, ADAM_RUNTIME_D256 * adam_rate)
            for multiplier in (0.75, 1.25, 1.5)
        ),
        *(
            Trial("muon", 768, multiplier, PROJECTED_MUON_RUNTIME_D768 * muon_rate)
            for multiplier in (0.25, 0.5)
        ),
    ]
    if retry_interrupted:
        return [
            Trial(
                trial.optimizer,
                trial.width,
                trial.lr_multiplier,
                trial.estimated_cost,
                suffix="-retry",
            )
            for trial in trials
            if (trial.optimizer, trial.lr_multiplier)
            in {("adamw", 0.75), ("adamw", 1.25), ("muon", 0.25)}
        ]
    if adam_lower:
        return [
            Trial(
                "adamw",
                256,
                multiplier,
                ADAM_RUNTIME_D256 * adam_rate,
                suffix="-lower",
            )
            for multiplier in (0.5, 0.625)
        ]
    if muon_upper:
        return [
            Trial(
                "muon",
                768,
                0.75,
                PROJECTED_MUON_RUNTIME_D768 * muon_rate,
                suffix="-upper",
            )
        ]
    if muon_one:
        return [
            Trial(
                "muon",
                768,
                1.0,
                PROJECTED_MUON_RUNTIME_D768 * muon_rate,
                suffix="-one",
            )
        ]
    if muon_midpoint:
        return [
            Trial(
                "muon",
                768,
                0.875,
                PROJECTED_MUON_RUNTIME_D768 * muon_rate,
                suffix="-midpoint",
            )
        ]
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
