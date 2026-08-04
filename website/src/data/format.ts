import type { BestRun } from "@/data/best-runs";

export const MODAL_B200_USD_PER_HOUR = 6.25;

export function formatModel(run: BestRun): string {
  return `d${run.dModel}`;
}

export function formatCompactNumber(value: number): string {
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 2,
  }).format(value);
}

export function formatDecimal(value: number, digits = 4): string {
  return value.toFixed(digits);
}

export function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

export function formatB200Cost(runtimeSec: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format((runtimeSec * MODAL_B200_USD_PER_HOUR) / 3600);
}
