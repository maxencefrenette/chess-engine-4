"""Command-line entrypoints for local training workflows."""

from __future__ import annotations

import argparse
from itertools import islice

from torch.utils.data import DataLoader

from chess_engine_4.data.leela import (
    DEFAULT_DATA_ENV_VAR,
    LeelaTarDataset,
    collate_records,
)

_DATA_HELP = f"Leela tar path, directory, or glob. Defaults to ${DEFAULT_DATA_ENV_VAR}."


def train() -> None:
    parser = argparse.ArgumentParser(description="Run a local smoke-test training loop.")
    parser.add_argument("--data", default=None, help=_DATA_HELP)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    dataset = LeelaTarDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_records)

    for step, batch in enumerate(islice(loader, args.steps), start=1):
        print(
            f"step={step} "
            f"planes={tuple(batch.planes.shape)} "
            f"policy={tuple(batch.policy.shape)} "
            f"value={tuple(batch.value.shape)}"
        )


def inspect_data() -> None:
    parser = argparse.ArgumentParser(description="Inspect Leela tar training records.")
    parser.add_argument("--data", default=None, help=_DATA_HELP)
    parser.add_argument("--records", type=int, default=5)
    args = parser.parse_args()

    dataset = LeelaTarDataset(args.data, max_records=args.records)
    for index, record in enumerate(dataset, start=1):
        print(
            f"record={index} "
            f"version={record.version} "
            f"input_format={record.input_format} "
            f"planes={record.planes.shape} "
            f"policy_sum={record.policy.sum():.4f} "
            f"value={record.value.tolist()} "
            f"plies_left={float(record.plies_left):.1f} "
            f"visits={record.visits}"
        )


def sample_batch() -> None:
    parser = argparse.ArgumentParser(description="Load and print one Leela training batch.")
    parser.add_argument("--data", default=None, help=_DATA_HELP)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    dataset = LeelaTarDataset(args.data, max_records=args.batch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_records)
    batch = next(iter(loader))
    print(f"planes: {tuple(batch.planes.shape)}")
    print(f"policy: {tuple(batch.policy.shape)}")
    print(f"value: {tuple(batch.value.shape)}")
    print(f"plies_left: {tuple(batch.plies_left.shape)}")
