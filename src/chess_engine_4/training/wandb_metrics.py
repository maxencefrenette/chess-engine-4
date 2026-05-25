"""Utilities for post-hoc W&B metric summaries."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from urllib.parse import urlparse

LOSS_KEY = "loss"
POLICY_TOP1_KEY = "metrics/policy_top1"
DEFAULT_TAIL = 100


@dataclass(frozen=True, slots=True)
class TailMetrics:
    wandb_url: str
    loss: float
    policy_top1: float
    tail_count: int


def wandb_tail_metrics() -> None:
    parser = argparse.ArgumentParser(
        description="Compute post-hoc tail means for W&B training metrics."
    )
    parser.add_argument("wandb_urls", nargs="*", help="W&B run URLs to summarize.")
    parser.add_argument("--csv", type=Path, default=None, help="CSV with a wandb_url column.")
    parser.add_argument("--tail", type=int, default=DEFAULT_TAIL)
    parser.add_argument("--timeout", type=int, default=60, help="W&B API timeout in seconds.")
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Rewrite --csv with loss and policy_top1 columns.",
    )
    args = parser.parse_args()

    urls = list(args.wandb_urls)
    rows: list[dict[str, str]] = []
    if args.csv is not None:
        rows = read_csv_rows(args.csv)
        urls.extend(row["wandb_url"] for row in rows)
    if not urls:
        parser.error("provide at least one W&B URL or --csv")

    metrics_by_url = fetch_tail_metrics_for_urls(urls, tail=args.tail, timeout=args.timeout)
    if args.write_csv:
        if args.csv is None:
            parser.error("--write-csv requires --csv")
        write_csv_rows(args.csv, rows, metrics_by_url)
    else:
        write_metrics_csv(metrics_by_url.values())


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(
    path: Path,
    rows: list[dict[str, str]],
    metrics_by_url: Mapping[str, TailMetrics],
) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    for fieldname in (
        "loss",
        "policy_top1",
        "tail_count",
    ):
        if fieldname not in fieldnames:
            fieldnames.append(fieldname)

    for row in rows:
        metrics = metrics_by_url[row["wandb_url"]]
        row["loss"] = str(metrics.loss)
        row["policy_top1"] = str(metrics.policy_top1)
        row["tail_count"] = str(metrics.tail_count)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_csv(metrics: Iterable[TailMetrics]) -> None:
    fieldnames = [
        "wandb_url",
        "loss",
        "policy_top1",
        "tail_count",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in metrics:
        writer.writerow(
            {
                "wandb_url": row.wandb_url,
                "loss": row.loss,
                "policy_top1": row.policy_top1,
                "tail_count": row.tail_count,
            }
        )


def fetch_tail_metrics_for_urls(
    urls: Iterable[str],
    *,
    tail: int,
    timeout: int = 60,
) -> dict[str, TailMetrics]:
    import wandb

    if tail <= 0:
        raise ValueError("tail must be positive.")

    api = wandb.Api(timeout=timeout)
    metrics_by_url: dict[str, TailMetrics] = {}
    for url in urls:
        if url in metrics_by_url:
            continue
        run = api.run(wandb_run_path_from_url(url))
        rows = list(run.scan_history(keys=["_step", LOSS_KEY, POLICY_TOP1_KEY], page_size=1000))
        metrics_by_url[url] = tail_metrics_from_history(url, rows, tail=tail)
    return metrics_by_url


def wandb_run_path_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    try:
        runs_index = parts.index("runs")
    except ValueError as exc:
        raise ValueError(f"not a W&B run URL: {url}") from exc
    if runs_index < 2 or runs_index + 1 >= len(parts):
        raise ValueError(f"not a W&B run URL: {url}")
    entity = parts[runs_index - 2]
    project = parts[runs_index - 1]
    run_id = parts[runs_index + 1]
    return f"{entity}/{project}/{run_id}"


def tail_metrics_from_history(
    wandb_url: str,
    rows: Iterable[Mapping[str, object]],
    *,
    tail: int,
) -> TailMetrics:
    sorted_rows = sorted(rows, key=lambda row: int(row.get("_step") or 0))
    pairs = tail_metric_pairs(sorted_rows, tail=tail)
    if not pairs:
        raise ValueError(
            f"{wandb_url} has no history rows with both {LOSS_KEY!r} and "
            f"{POLICY_TOP1_KEY!r} values."
        )
    loss_values = [loss for loss, _ in pairs]
    policy_top1_values = [policy_top1 for _, policy_top1 in pairs]
    return TailMetrics(
        wandb_url=wandb_url,
        loss=fmean(loss_values),
        policy_top1=fmean(policy_top1_values),
        tail_count=len(pairs),
    )


def tail_metric_pairs(
    rows: Iterable[Mapping[str, object]],
    *,
    tail: int,
) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        loss = row.get(LOSS_KEY)
        policy_top1 = row.get(POLICY_TOP1_KEY)
        if isinstance(loss, int | float) and isinstance(policy_top1, int | float):
            pairs.append((float(loss), float(policy_top1)))
    return pairs[-tail:]


def tail_values(
    rows: Iterable[Mapping[str, object]],
    key: str,
    *,
    tail: int,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, int | float):
            values.append(float(value))
    return values[-tail:]
