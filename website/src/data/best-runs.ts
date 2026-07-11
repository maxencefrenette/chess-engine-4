import fs from "node:fs";
import path from "node:path";

export type ModelFamilyId = "mlp" | "mlp_moe16a2";

export type BestRun = {
  budget: string;
  sourceExperiment: string;
  modelKind: string;
  runName: string;
  wandbUrl: string;
  compute: number;
  physicalFlops: number;
  dModel: number;
  depth: number;
  batchSize: number;
  lr: number;
  params: number;
  samplesSeen: number;
  samplesPerParam: number;
  loss: number;
  lossUpper1sd: number;
  policyTop1: number;
  runtimeSec: number;
};

export type ExtrapolatedRun = {
  budget: string;
  compute: number;
  physicalFlops: number;
  params: number;
  samplesSeen: number;
  samplesPerParam: number;
  lossUpper1sd: number;
  policyTop1: number;
};

export type CurvePoint = {
  compute: number;
  physicalFlops: number;
  value: number;
};

export type ScalingFamily = {
  id: ModelFamilyId;
  name: string;
  description: string;
  observed: BestRun[];
  extrapolated: ExtrapolatedRun[];
  curves: {
    lossScore: CurvePoint[];
    policyTop1: CurvePoint[];
    params: CurvePoint[];
    samples: CurvePoint[];
    samplesPerParam: CurvePoint[];
  };
};

export type ModelFamily = Pick<ScalingFamily, "id" | "name" | "description">;

type ScalingData = {
  version: number;
  families: Record<ModelFamilyId, ScalingFamily>;
};

let cachedData: ScalingData | undefined;

export const modelFamilies: ModelFamily[] = Object.values(readScalingData().families).map(
  ({ id, name, description }) => ({ id, name, description }),
);

export function getFamily(id: string): ModelFamily | undefined {
  return modelFamilies.find((family) => family.id === id);
}

export function readScalingFamily(family: ModelFamily): ScalingFamily {
  return readScalingData().families[family.id];
}

export function readBestRuns(family: ModelFamily): BestRun[] {
  return readScalingFamily(family).observed;
}

export function latestRun(runs: BestRun[]): BestRun | undefined {
  return runs.at(-1);
}

function readScalingData(): ScalingData {
  if (cachedData) return cachedData;
  const generatedPath = path.join(process.cwd(), "src/generated/scaling-laws.json");
  cachedData = JSON.parse(fs.readFileSync(generatedPath, "utf8")) as ScalingData;
  return cachedData;
}
