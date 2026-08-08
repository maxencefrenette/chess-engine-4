import scalingData from "@/generated/scaling-laws.json";

export type TrainingGpu = "B200" | "RTX-PRO-6000";

export type BestRun = {
  name: string;
  status: "current" | "stale";
  runName: string;
  wandbUrl: string;
  physicalFlops: number;
  depth: number;
  batchSize: number;
  lr: number;
  params: number;
  samplesSeen: number;
  loss: number;
  policyTop1: number;
  gpu: TrainingGpu;
  runtimeSec: number;
};

export type ExtrapolatedRun = {
  name: string;
  physicalFlops: number;
  params: number;
  samplesSeen: number;
  loss: number;
  policyTop1: number;
  lr: number;
  batchSize: number;
};

export type CurvePoint = {
  physicalFlops: number;
  value: number;
};

export type ScalingFamily = {
  id: string;
  name: string;
  description: string;
  trainingRatio: number;
  runs: BestRun[];
  extrapolated: ExtrapolatedRun[];
  curves: {
    loss: CurvePoint[];
    policyTop1: CurvePoint[];
    params: CurvePoint[];
    samples: CurvePoint[];
    samplesPerParam: CurvePoint[];
    lr: CurvePoint[];
    steps: CurvePoint[];
    batchSize: CurvePoint[];
  };
};

type ScalingData = {
  families: ScalingFamily[];
};

export const modelFamilies = readScalingData().families;

export function getFamily(id: string): ScalingFamily | undefined {
  return modelFamilies.find((family) => family.id === id);
}

export function currentRuns(family: ScalingFamily): BestRun[] {
  return family.runs.filter((run) => run.status === "current");
}

export function latestRun(runs: BestRun[]): BestRun | undefined {
  return runs.at(-1);
}

function readScalingData(): ScalingData {
  return scalingData as ScalingData;
}
