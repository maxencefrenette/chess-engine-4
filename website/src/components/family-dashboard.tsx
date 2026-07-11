"use client";

import { useState } from "react";
import { LineChart } from "@/components/line-chart";
import type { CurvePoint, ExtrapolatedRun, ScalingFamily } from "@/data/best-runs";

type XAxisMode = "compute" | "flops";
type MetricKey =
  | "loss"
  | "policyTop1"
  | "params"
  | "samplesSeen"
  | "samplesPerParam"
  | "lr";

export function FamilyDashboard({ family }: { family: ScalingFamily }) {
  const [xAxisMode, setXAxisMode] = useState<XAxisMode>("compute");
  const xLabel = xAxisMode === "compute" ? "Compute budget" : "Physical FLOPs";
  const xValue = (point: { compute: number; physicalFlops: number }) =>
    xAxisMode === "compute" ? point.compute : point.physicalFlops;

  return (
    <>
      <section className="mt-5 flex items-end justify-between rounded-lg border border-zinc-200 bg-white px-5 py-4 shadow-sm">
        <div>
          <div className="mb-2 text-xs font-medium uppercase text-zinc-500">X-axis</div>
          <div className="inline-flex rounded-md border border-zinc-200 bg-zinc-100 p-1">
            <ModeButton
              active={xAxisMode === "compute"}
              onClick={() => setXAxisMode("compute")}
            >
              Compute budget
            </ModeButton>
            <ModeButton active={xAxisMode === "flops"} onClick={() => setXAxisMode("flops")}>
              Physical FLOPs
            </ModeButton>
          </div>
        </div>
        <div className="flex items-center gap-5 pb-1 text-sm text-zinc-600">
          <LegendMarker filled label="Observed" />
          <LegendMarker label="Extrapolated" />
        </div>
      </section>

      <section className="mt-5 grid grid-cols-2 gap-5">
        <ChartCard
          title="Loss"
          yLabel="Loss"
          observed={metricPoints(family, "loss", xValue)}
          extrapolated={extrapolatedPoints(family.extrapolated, "loss", xValue)}
          curve={curvePoints(family.curves.loss, xValue)}
          xLabel={xLabel}
        />
        <ChartCard
          title="Policy top-1"
          yLabel="Top-1 accuracy"
          observed={metricPoints(family, "policyTop1", xValue)}
          extrapolated={extrapolatedPoints(family.extrapolated, "policyTop1", xValue)}
          curve={curvePoints(family.curves.policyTop1, xValue)}
          stroke="#16a34a"
          valueFormat="percent"
          xLabel={xLabel}
        />
        <ChartCard
          title="Parameters"
          yLabel="Total parameters"
          observed={metricPoints(family, "params", xValue)}
          extrapolated={extrapolatedPoints(family.extrapolated, "params", xValue)}
          curve={curvePoints(family.curves.params, xValue)}
          stroke="#9333ea"
          valueFormat="compact"
          xLabel={xLabel}
          yScale="log"
        />
        <ChartCard
          title="Samples seen"
          yLabel="Training samples"
          observed={metricPoints(family, "samplesSeen", xValue)}
          extrapolated={extrapolatedPoints(family.extrapolated, "samplesSeen", xValue)}
          curve={curvePoints(family.curves.samples, xValue)}
          stroke="#ea580c"
          valueFormat="compact"
          xLabel={xLabel}
          yScale="log"
        />
        <ChartCard
          title="Samples per parameter"
          yLabel="Samples / parameter"
          observed={metricPoints(family, "samplesPerParam", xValue)}
          extrapolated={extrapolatedPoints(family.extrapolated, "samplesPerParam", xValue)}
          curve={curvePoints(family.curves.samplesPerParam, xValue)}
          stroke="#0891b2"
          xLabel={xLabel}
        />
        <ChartCard
          title="Learning rate"
          yLabel="Learning rate"
          observed={metricPoints(family, "lr", xValue)}
          extrapolated={extrapolatedPoints(family.extrapolated, "lr", xValue)}
          curve={curvePoints(family.curves.lr, xValue)}
          stroke="#dc2626"
          valueFormat="scientific"
          xLabel={xLabel}
          yScale="log"
        />
      </section>
    </>
  );
}

function ModeButton({
  active,
  children,
  onClick,
}: {
  active: boolean;
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      className={`rounded px-3 py-1.5 text-sm font-medium transition ${active ? "bg-white text-zinc-950 shadow-sm" : "text-zinc-500 hover:text-zinc-800"}`}
      onClick={onClick}
      type="button"
    >
      {children}
    </button>
  );
}

function LegendMarker({ filled = false, label }: { filled?: boolean; label: string }) {
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
  observed: { x: number; y: number; budget: string }[];
  extrapolated: { x: number; y: number; budget: string }[];
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

function metricPoints(
  family: ScalingFamily,
  metric: MetricKey,
  xValue: (point: { compute: number; physicalFlops: number }) => number,
) {
  return family.observed.map((run) => ({ x: xValue(run), y: run[metric], budget: run.budget }));
}

function extrapolatedPoints(
  runs: ExtrapolatedRun[],
  metric: MetricKey,
  xValue: (point: { compute: number; physicalFlops: number }) => number,
) {
  return runs.map((run) => ({ x: xValue(run), y: run[metric], budget: run.budget }));
}

function curvePoints(
  points: CurvePoint[],
  xValue: (point: { compute: number; physicalFlops: number }) => number,
) {
  return points.map((point) => ({ x: xValue(point), y: point.value }));
}
