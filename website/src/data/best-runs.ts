import scalingData from "@/generated/scaling-laws.json";

export type ModelFamilyId = "dense" | "moe64a2";
export type TrainingGpu = "B200" | "RTX-PRO-6000";

export type BestRun = {
  name: string;
  sourceExperiment: string;
  modelKind: string;
  runName: string;
  wandbUrl: string;
  physicalFlops: number;
  dModel: number;
  trainingRatio: number;
  depth: number;
  batchSize: number;
  steps: number;
  lr: number;
  params: number;
  samplesSeen: number;
  samplesPerParam: number;
  loss: number;
  policyTop1: number;
  gpu: TrainingGpu;
  runtimeSec: number;
  stale: boolean;
};

export type ExtrapolatedRun = {
  name: string;
  physicalFlops: number;
  params: number;
  samplesSeen: number;
  samplesPerParam: number;
  loss: number;
  policyTop1: number;
  lr: number;
  steps: number;
  batchSize: number;
};

export type CurvePoint = {
  physicalFlops: number;
  value: number;
};

export type ScalingFamily = {
  id: ModelFamilyId;
  name: string;
  description: string;
  trainingRatio: number;
  observed: BestRun[];
  staleObserved: BestRun[];
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

export type ModelFamily = Pick<ScalingFamily, "id" | "name" | "description">;

type ScalingData = {
  families: Record<ModelFamilyId, ScalingFamily>;
};

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
  return scalingData as ScalingData;
}
