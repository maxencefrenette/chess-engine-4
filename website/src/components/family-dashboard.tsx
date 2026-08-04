"use client";

import { LineChart } from "@/components/line-chart";
import type { CurvePoint, ExtrapolatedRun, ScalingFamily } from "@/data/best-runs";

type MetricKey =
  | "loss"
  | "policyTop1"
  | "params"
  | "samplesSeen"
  | "samplesPerParam"
  | "lr"
  | "steps"
  | "batchSize";

export function FamilyDashboard({ family }: { family: ScalingFamily }) {
  const xLabel = "Training FLOPs";

  return (
    <>
      <section className="mt-5 flex justify-end">
        <div className="flex items-center gap-5 text-sm text-zinc-600">
          <LegendMarker filled label="Observed" />
          <LegendMarker label="Extrapolated" />
        </div>
      </section>

      <section className="mt-5 grid grid-cols-2 gap-5">
        <ChartCard
          title="Loss"
          yLabel="Loss"
          observed={metricPoints(family, "loss")}
          extrapolated={extrapolatedPoints(family.extrapolated, "loss")}
          curve={curvePoints(family.curves.loss)}
          xLabel={xLabel}
        />
        <ChartCard
          title="Policy top-1"
          yLabel="Top-1 accuracy"
          observed={metricPoints(family, "policyTop1")}
          extrapolated={extrapolatedPoints(family.extrapolated, "policyTop1")}
          curve={curvePoints(family.curves.policyTop1)}
          stroke="#16a34a"
          valueFormat="percent"
          xLabel={xLabel}
        />
        <ChartCard
          title="Parameters"
          yLabel="Total parameters"
          observed={metricPoints(family, "params")}
          extrapolated={extrapolatedPoints(family.extrapolated, "params")}
          curve={curvePoints(family.curves.params)}
          stroke="#9333ea"
          valueFormat="compact"
          xLabel={xLabel}
          yScale="log"
        />
        <ChartCard
          title="Samples seen"
          yLabel="Training samples"
          observed={metricPoints(family, "samplesSeen")}
          extrapolated={extrapolatedPoints(family.extrapolated, "samplesSeen")}
          curve={curvePoints(family.curves.samples)}
          stroke="#ea580c"
          valueFormat="compact"
          xLabel={xLabel}
          yScale="log"
        />
        <ChartCard
          title="Samples per parameter"
          yLabel="Samples / parameter"
          observed={metricPoints(family, "samplesPerParam")}
          extrapolated={extrapolatedPoints(family.extrapolated, "samplesPerParam")}
          curve={curvePoints(family.curves.samplesPerParam)}
          stroke="#0891b2"
          xLabel={xLabel}
        />
        <ChartCard
          title="Learning rate"
          yLabel="Learning rate"
          observed={metricPoints(family, "lr")}
          extrapolated={extrapolatedPoints(family.extrapolated, "lr")}
          curve={curvePoints(family.curves.lr)}
          stroke="#dc2626"
          valueFormat="scientific"
          xLabel={xLabel}
          yScale="log"
        />
        <ChartCard
          title="Steps"
          yLabel="Optimization steps"
          observed={metricPoints(family, "steps")}
          extrapolated={extrapolatedPoints(family.extrapolated, "steps")}
          curve={curvePoints(family.curves.steps)}
          stroke="#ca8a04"
          valueFormat="compact"
          xLabel={xLabel}
          yScale="log"
        />
        <ChartCard
          title="Batch size"
          yLabel="Batch size"
          observed={metricPoints(family, "batchSize")}
          extrapolated={extrapolatedPoints(family.extrapolated, "batchSize")}
          curve={curvePoints(family.curves.batchSize)}
          stroke="#db2777"
          valueFormat="compact"
          xLabel={xLabel}
          yScale="log"
        />
      </section>
    </>
  );
}

function LegendMarker({
  filled = false,
  label,
}: {
  filled?: boolean;
  label: string;
}) {
  return (
    <span className="flex items-center gap-2">
      <span
        className={`size-3 rounded-full border-2 border-blue-600 ${filled ? "bg-blue-600" : "bg-white"}`}
      />
      {label}
    </span>
  );
}

function ChartCard({
  className = "",
  title,
  yLabel,
  observed,
  extrapolated,
  curve,
  stroke,
  valueFormat,
  xLabel,
  yScale,
}: {
  className?: string;
  title: string;
  yLabel: string;
  observed: { x: number; y: number; name: string }[];
  extrapolated: { x: number; y: number; name: string }[];
  curve: { x: number; y: number }[];
  stroke?: string;
  valueFormat?: "decimal" | "percent" | "compact" | "scientific";
  xLabel: string;
  yScale?: "linear" | "log";
}) {
  return (
    <div className={`rounded-lg border border-zinc-200 bg-white p-5 shadow-sm ${className}`}>
      <h2 className="text-lg font-semibold text-zinc-950">{title}</h2>
      <div className="mt-4">
        <LineChart
          extrapolatedPoints={extrapolated}
          fitPoints={curve}
          label={title}
          points={observed}
          stroke={stroke}
          valueFormat={valueFormat}
          xLabel={xLabel}
          yLabel={yLabel}
          yScale={yScale}
        />
      </div>
    </div>
  );
}

function metricPoints(family: ScalingFamily, metric: MetricKey) {
  return family.observed.map((run) => ({
    x: run.physicalFlops,
    y: run[metric],
    name: run.name,
  }));
}

function extrapolatedPoints(
  runs: ExtrapolatedRun[],
  metric: MetricKey,
) {
  return runs.map((run) => ({ x: run.physicalFlops, y: run[metric], name: run.name }));
}

function curvePoints(points: CurvePoint[]) {
  return points.map((point) => ({ x: point.physicalFlops, y: point.value }));
}
