import type { BestRun, TrainingGpu } from "@/data/best-runs";

const MODAL_GPU_USD_PER_SECOND: Record<TrainingGpu, number> = {
  B200: 0.001736,
  "RTX-PRO-6000": 0.000842,
};
const MODAL_CPU_USD_PER_CORE_SECOND = 0.0000131;
const TRAINING_CPU_CORES = 8;

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

export function formatGpu(gpu: TrainingGpu): string {
  return gpu === "RTX-PRO-6000" ? "RTX PRO 6000" : gpu;
}

export function trainingCost(run: BestRun): number {
  const rate =
    MODAL_GPU_USD_PER_SECOND[run.gpu] +
    TRAINING_CPU_CORES * MODAL_CPU_USD_PER_CORE_SECOND;
  return run.runtimeSec * rate;
}

export function formatTrainingCost(run: BestRun): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(trainingCost(run));
}
