"""Command-line entrypoints for local training workflows."""

from __future__ import annotations

import argparse
from itertools import islice

from chess_engine_4.data.leela import (
    DEFAULT_DATA_ENV_VAR,
    LeelaTarDataset,
)

_DATA_HELP = f"Leela tar path, directory, or glob. Defaults to ${DEFAULT_DATA_ENV_VAR}."


def train() -> None:
    parser = argparse.ArgumentParser(description="Run a local smoke-test training loop.")
    parser.add_argument("--data", default=None, help=_DATA_HELP)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    dataset = LeelaTarDataset(args.data, batch_size=args.batch_size)

    for step, (planes, policy, value) in enumerate(islice(dataset, args.steps), start=1):
        print(
            f"step={step} "
            f"planes={tuple(planes.shape)} "
            f"policy={tuple(policy.shape)} "
            f"value={tuple(value.shape)}"
        )


def inspect_data() -> None:
    parser = argparse.ArgumentParser(description="Inspect Leela tar training records.")
    parser.add_argument("--data", default=None, help=_DATA_HELP)
    parser.add_argument("--records", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    dataset = LeelaTarDataset(args.data, batch_size=args.batch_size, max_records=args.records)
    seen = 0
    for planes, policy, value in dataset:
        batch_size = planes.shape[0]
        legal = policy >= 0
        legal_policy_sum = policy.masked_fill(~legal, 0).sum(dim=1)
        illegal_count = (~legal).sum(dim=1)
        seen += batch_size
        print(
            f"records={seen} "
            f"planes={tuple(planes.shape)} "
            f"policy={tuple(policy.shape)} "
            f"value={tuple(value.shape)} "
            f"legal_policy_sum=[{legal_policy_sum.min():.4f}, {legal_policy_sum.max():.4f}] "
            f"illegal_moves=[{illegal_count.min()}, {illegal_count.max()}] "
            f"plies_left=[{value[:, 0, 2].min():.1f}, {value[:, 0, 2].max():.1f}]"
        )


def sample_batch() -> None:
    parser = argparse.ArgumentParser(description="Load and print one Leela training batch.")
    parser.add_argument("--data", default=None, help=_DATA_HELP)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    dataset = LeelaTarDataset(args.data, batch_size=args.batch_size, max_records=args.batch_size)
    planes, policy, value = next(iter(dataset))
    print(f"planes: {tuple(planes.shape)}")
    print(f"policy: {tuple(policy.shape)}")
    print(f"value: {tuple(value.shape)}")
