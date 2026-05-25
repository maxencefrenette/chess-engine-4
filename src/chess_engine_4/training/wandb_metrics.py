"""Utilities for W&B selection metric summaries."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

LOSS_KEY = "loss/task[ema=0.99]"
POLICY_TOP1_KEY = "metrics/policy_top1[ema=0.99]"


@dataclass(frozen=True, slots=True)
class WandbMetrics:
    wandb_url: str
    loss: float
    policy_top1: float


def wandb_metrics() -> None:
    parser = argparse.ArgumentParser(
        description="Read smoothed selection metrics from W&B run summaries."
    )
    parser.add_argument("wandb_urls", nargs="*", help="W&B run URLs to summarize.")
    parser.add_argument("--csv", type=Path, default=None, help="CSV with a wandb_url column.")
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

    metrics_by_url = fetch_metrics_for_urls(urls, timeout=args.timeout)
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
    metrics_by_url: Mapping[str, WandbMetrics],
) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    for fieldname in (
        "loss",
        "policy_top1",
    ):
        if fieldname not in fieldnames:
            fieldnames.append(fieldname)

    for row in rows:
        metrics = metrics_by_url[row["wandb_url"]]
        row["loss"] = str(metrics.loss)
        row["policy_top1"] = str(metrics.policy_top1)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_csv(metrics: Iterable[WandbMetrics]) -> None:
    fieldnames = [
        "wandb_url",
        "loss",
        "policy_top1",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in metrics:
        writer.writerow(
            {
                "wandb_url": row.wandb_url,
                "loss": row.loss,
                "policy_top1": row.policy_top1,
            }
        )


def fetch_metrics_for_urls(
    urls: Iterable[str],
    *,
    timeout: int = 60,
) -> dict[str, WandbMetrics]:
    import wandb

    api = wandb.Api(timeout=timeout)
    metrics_by_url: dict[str, WandbMetrics] = {}
    for url in urls:
        if url in metrics_by_url:
            continue
        run = api.run(wandb_run_path_from_url(url))
        metrics_by_url[url] = metrics_from_summary(url, run.summary)
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


def metrics_from_summary(
    wandb_url: str,
    summary: Mapping[str, object],
) -> WandbMetrics:
    loss = summary.get(LOSS_KEY)
    policy_top1 = summary.get(POLICY_TOP1_KEY)
    if not isinstance(loss, int | float) or not isinstance(policy_top1, int | float):
        raise ValueError(
            f"{wandb_url} summary has no {LOSS_KEY!r} and {POLICY_TOP1_KEY!r} values."
        )
    return WandbMetrics(
        wandb_url=wandb_url,
        loss=float(loss),
        policy_top1=float(policy_top1),
    )
