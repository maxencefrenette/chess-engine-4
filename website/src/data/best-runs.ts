import fs from "node:fs";
import path from "node:path";
import { parse } from "smol-toml";

export type ModelFamilyId = "mlp" | "mlp_moe16a2";

export type BestRun = {
  budget: string;
  sourceExperiment: string;
  modelKind: string;
  runName: string;
  wandbUrl: string;
  compute: number;
  dModel: number;
  depth: number;
  batchSize: number;
  lr: number;
  params: number;
  samplesSeen: number;
  loss: number;
  lossUpper1sd?: number;
  policyTop1: number;
  runtimeSec: number;
};

export type ModelFamily = {
  id: ModelFamilyId;
  name: string;
  description: string;
  bestRunsPath: string;
};

export const modelFamilies: ModelFamily[] = [
  {
    id: "mlp",
    name: "Dense MLP",
    description: "Single-token dense SwiGLU MLP trained on lc0 planes.",
    bestRunsPath: "experiments/best-runs-dense.toml",
  },
  {
    id: "mlp_moe16a2",
    name: "MLP-MoE 16a2",
    description: "Single-token MoE with 16 experts and 2 active experts.",
    bestRunsPath: "experiments/best-runs-mlp_moe16a2.toml",
  },
];

const repoRoot = path.resolve(process.cwd(), "..");

export function getFamily(id: string): ModelFamily | undefined {
  return modelFamilies.find((family) => family.id === id);
}

export function readBestRuns(family: ModelFamily): BestRun[] {
  const raw = fs.readFileSync(path.join(repoRoot, family.bestRunsPath), "utf8");
  const parsed = parse(raw) as {
    runs?: Record<string, Record<string, unknown>>;
  };

  return Object.entries(parsed.runs ?? {})
    .map(([budget, run]) => normalizeRun(budget, run))
    .sort((a, b) => a.compute - b.compute);
}

export function latestRun(runs: BestRun[]): BestRun | undefined {
  return runs.at(-1);
}

export function formatModel(run: BestRun): string {
  return `d${run.dModel}x${run.depth}`;
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

function normalizeRun(budget: string, run: Record<string, unknown>): BestRun {
  return {
    budget,
    sourceExperiment: stringValue(run.source_experiment),
    modelKind: stringValue(run.model_kind),
    runName: stringValue(run.run_name),
    wandbUrl: stringValue(run.wandb_url),
    compute: numberValue(run.compute),
    dModel: numberValue(run.d_model),
    depth: numberValue(run.depth),
    batchSize: numberValue(run.batch_size),
    lr: numberValue(run.lr),
    params: numberValue(run.params),
    samplesSeen: numberValue(run.samples_seen),
    loss: numberValue(run.loss),
    lossUpper1sd:
      run.loss_upper_1sd === undefined ? undefined : numberValue(run.loss_upper_1sd),
    policyTop1: numberValue(run.policy_top1),
    runtimeSec: numberValue(run.runtime_sec),
  };
}

function numberValue(value: unknown): number {
  if (typeof value !== "number") {
    throw new TypeError(`Expected number, got ${typeof value}`);
  }
  return value;
}

function stringValue(value: unknown): string {
  if (typeof value !== "string") {
    throw new TypeError(`Expected string, got ${typeof value}`);
  }
  return value;
}
