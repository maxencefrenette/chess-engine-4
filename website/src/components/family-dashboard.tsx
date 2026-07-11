"use client";

import { useState } from "react";
import { LineChart } from "@/components/line-chart";
import type { BestRun } from "@/data/best-runs";
import { formatCompactNumber, formatDecimal, formatPercent } from "@/data/format";

type XAxisMode = "compute" | "flops";

export function FamilyDashboard({ runs }: { runs: BestRun[] }) {
  const [xAxisMode, setXAxisMode] = useState<XAxisMode>("compute");
  const [showTarget, setShowTarget] = useState(true);
  const [targetText, setTargetText] = useState("1e23");
  const parsedTarget = Number(targetText);
  const targetCompute = Number.isFinite(parsedTarget) && parsedTarget > 0 ? parsedTarget : null;
  const xValue = (run: BestRun) => (xAxisMode === "compute" ? run.compute : physicalFlops(run));
  const targetX =
    showTarget && targetCompute
      ? xAxisMode === "compute"
        ? targetCompute
        : predict(runs, targetCompute, physicalFlops, true)
      : undefined;
  const projection = showTarget && targetCompute ? projectTarget(runs, targetCompute) : null;

  return (
    <>
      <section className="mt-5 rounded-lg border border-zinc-200 bg-white px-5 py-4 shadow-sm">
        <div className="flex items-center justify-between gap-8">
          <div className="flex items-center gap-6">
            <div>
              <div className="mb-2 text-xs font-medium uppercase text-zinc-500">X-axis</div>
              <div className="inline-flex rounded-md border border-zinc-200 bg-zinc-100 p-1">
                <ModeButton
                  active={xAxisMode === "compute"}
                  onClick={() => setXAxisMode("compute")}
                >
                  Compute budget
                </ModeButton>
                <ModeButton
                  active={xAxisMode === "flops"}
                  onClick={() => setXAxisMode("flops")}
                >
                  Physical FLOPs
                </ModeButton>
              </div>
            </div>
            <label className="flex items-center gap-2 pt-5 text-sm font-medium text-zinc-700">
              <input
                checked={showTarget}
                className="size-4 accent-blue-600"
                onChange={(event) => setShowTarget(event.target.checked)}
                type="checkbox"
              />
              Extrapolate
            </label>
            <label className="block">
              <span className="mb-2 block text-xs font-medium uppercase text-zinc-500">
                Target compute budget
              </span>
              <input
                aria-invalid={showTarget && targetCompute === null}
                className="h-9 w-32 rounded-md border border-zinc-300 bg-white px-3 font-mono text-sm text-zinc-950 outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-100 disabled:bg-zinc-100 disabled:text-zinc-400"
                disabled={!showTarget}
                onChange={(event) => setTargetText(event.target.value)}
                spellCheck={false}
                value={targetText}
              />
            </label>
          </div>
          <p className="max-w-sm text-right text-xs leading-5 text-zinc-500">
            Physical FLOPs are derived from observed compute, batch size, and step count.
          </p>
        </div>

        {projection ? (
          <dl className="mt-4 grid grid-cols-5 divide-x divide-zinc-200 border-t border-zinc-200 pt-4 tabular-nums">
            <Projection label="Loss score" value={formatDecimal(projection.loss)} />
            <Projection label="Policy top-1" value={formatPercent(projection.policyTop1)} />
            <Projection label="Parameters" value={formatCompactNumber(projection.params)} />
            <Projection label="Samples" value={formatCompactNumber(projection.samples)} />
            <Projection label="Samples / param" value={formatDecimal(projection.samples / projection.params, 1)} />
          </dl>
        ) : null}
      </section>

      <section className="mt-5 grid grid-cols-2 gap-5">
        <ChartCard
          title="Loss score"
          yLabel="Loss upper 1 SD"
          points={runs.map((run) => ({
            x: xValue(run),
            y: run.lossUpper1sd ?? run.loss,
            budget: run.budget,
          }))}
          targetX={targetX}
          xLabel={xAxisMode === "compute" ? "Compute budget" : "Physical FLOPs"}
        />
        <ChartCard
          title="Policy top-1"
          yLabel="Top-1 accuracy"
          points={runs.map((run) => ({ x: xValue(run), y: run.policyTop1, budget: run.budget }))}
          stroke="#16a34a"
          targetX={targetX}
          valueFormat="percent"
          xLabel={xAxisMode === "compute" ? "Compute budget" : "Physical FLOPs"}
        />
        <ChartCard
          title="Parameters"
          yLabel="Total parameters"
          points={runs.map((run) => ({ x: xValue(run), y: run.params, budget: run.budget }))}
          stroke="#9333ea"
          targetX={targetX}
          valueFormat="compact"
          xLabel={xAxisMode === "compute" ? "Compute budget" : "Physical FLOPs"}
          yScale="log"
        />
        <ChartCard
          title="Samples seen"
          yLabel="Training samples"
          points={runs.map((run) => ({ x: xValue(run), y: run.samplesSeen, budget: run.budget }))}
          stroke="#ea580c"
          targetX={targetX}
          valueFormat="compact"
          xLabel={xAxisMode === "compute" ? "Compute budget" : "Physical FLOPs"}
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

function Projection({ label, value }: { label: string; value: string }) {
  return (
    <div className="px-4 first:pl-0">
      <dt className="text-xs font-medium uppercase text-zinc-500">{label}</dt>
      <dd className="mt-1 text-lg font-semibold text-zinc-950">{value}</dd>
    </div>
  );
}

function ChartCard({
  title,
  yLabel,
  points,
  stroke,
  targetX,
  valueFormat,
  xLabel,
  yScale,
}: {
  title: string;
  yLabel: string;
  points: { x: number; y: number; budget: string }[];
  stroke?: string;
  targetX?: number;
  valueFormat?: "decimal" | "percent" | "compact";
  xLabel: string;
  yScale?: "linear" | "log";
}) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-5 shadow-sm">
      <h2 className="text-lg font-semibold text-zinc-950">{title}</h2>
      <div className="mt-4">
        <LineChart
          label={title}
          points={points}
          stroke={stroke}
          targetX={targetX}
          valueFormat={valueFormat}
          xLabel={xLabel}
          yLabel={yLabel}
          yScale={yScale}
        />
      </div>
    </div>
  );
}

function physicalFlops(run: BestRun): number {
  const steps = run.samplesSeen / run.batchSize;
  return run.compute / steps;
}

function projectTarget(runs: BestRun[], targetCompute: number) {
  const loss = predict(runs, targetCompute, (run) => run.lossUpper1sd ?? run.loss, false);
  const policyTop1 = predict(runs, targetCompute, (run) => run.policyTop1, false);
  const params = predict(runs, targetCompute, (run) => run.params, true);
  const samples = predict(runs, targetCompute, (run) => run.samplesSeen, true);
  return { loss, policyTop1, params, samples };
}

function predict(
  runs: BestRun[],
  targetCompute: number,
  value: (run: BestRun) => number,
  logY: boolean,
): number {
  const xs = runs.map((run) => Math.log10(run.compute));
  const ys = runs.map((run) => (logY ? Math.log10(value(run)) : value(run)));
  const xMean = mean(xs);
  const yMean = mean(ys);
  const covariance = xs.reduce((sum, x, index) => sum + (x - xMean) * (ys[index] - yMean), 0);
  const variance = xs.reduce((sum, x) => sum + (x - xMean) ** 2, 0);
  const slope = variance === 0 ? 0 : covariance / variance;
  const prediction = yMean + slope * (Math.log10(targetCompute) - xMean);
  return logY ? 10 ** prediction : prediction;
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
