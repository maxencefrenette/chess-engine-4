"use client";

import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type LineChartPoint = {
  x: number;
  y: number;
  budget: string;
};

type LineChartProps = {
  points: LineChartPoint[];
  label: string;
  xLabel: string;
  yLabel: string;
  stroke?: string;
  valueFormat?: "decimal" | "percent" | "compact";
  yScale?: "linear" | "log";
  targetX?: number;
};

type ChartPoint = {
  budget: string;
  logX: number;
  plotY?: number;
  x: number;
  y?: number;
  fit?: number;
};

export function LineChart({
  points,
  label,
  xLabel,
  yLabel,
  stroke = "#2563eb",
  valueFormat = "decimal",
  yScale = "linear",
  targetX,
}: LineChartProps) {
  if (points.length === 0) {
    return null;
  }

  const fit = linearFit(points, yScale);
  const chartPoints: ChartPoint[] = points.map((point) => ({
    ...point,
    logX: Math.log10(point.x),
    plotY: plotY(point.y, yScale),
    fit: fit(Math.log10(point.x)),
  }));
  if (targetX && targetX > Math.max(...points.map((point) => point.x))) {
    chartPoints.push({
      budget: "Target",
      x: targetX,
      logX: Math.log10(targetX),
      fit: fit(Math.log10(targetX)),
    });
  }
  const yValues = chartPoints.flatMap((point) =>
    [point.plotY, point.fit].filter((value): value is number => value !== undefined),
  );
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  const yPadding = Math.max((yMax - yMin) * 0.06, Math.abs(yMin) * 0.01, 0.01);

  return (
    <div aria-label={label} className="h-[360px] w-full" role="img">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={chartPoints} margin={{ top: 12, right: 20, bottom: 18, left: 12 }}>
          <CartesianGrid stroke="#e4e4e7" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="logX"
            domain={["dataMin", "dataMax"]}
            label={{ value: xLabel, position: "insideBottom", offset: -10 }}
            tickFormatter={(value: number) => `1e${Math.round(value)}`}
            type="number"
          />
          <YAxis
            dataKey="plotY"
            domain={[yMin - yPadding, yMax + yPadding]}
            label={{ value: yLabel, angle: -90, position: "insideLeft" }}
            tickFormatter={(value: number) =>
              formatValue(yScale === "log" ? 10 ** value : value, valueFormat)
            }
            width={68}
          />
          <Tooltip
            cursor={{ stroke: "#a1a1aa", strokeDasharray: "3 3" }}
            content={({ active, payload }) => {
              const point = payload?.find((entry) => entry.dataKey === "plotY")?.payload as
                | ChartPoint
                | undefined;
              if (!active || !point || point.y === undefined) return null;
              return (
                <div className="rounded-md border border-zinc-200 bg-white px-3 py-2 text-sm shadow-lg">
                  <div className="font-semibold text-zinc-950">{point.budget}</div>
                  <div className="mt-1 text-zinc-600">
                    {yLabel}: {formatValue(point.y, valueFormat, true)}
                  </div>
                </div>
              );
            }}
          />
          {targetX ? (
            <ReferenceLine
              label={{ value: "Target", fill: "#71717a", fontSize: 11, position: "insideTopRight" }}
              stroke="#a1a1aa"
              strokeDasharray="3 3"
              x={Math.log10(targetX)}
            />
          ) : null}
          <Line
            dataKey="fit"
            dot={false}
            isAnimationActive={false}
            stroke={stroke}
            strokeDasharray="5 5"
            strokeOpacity={0.65}
            strokeWidth={1.5}
            type="linear"
          />
          <Scatter dataKey="plotY" fill={stroke} isAnimationActive={false} r={5} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function linearFit(
  points: LineChartPoint[],
  yScale: NonNullable<LineChartProps["yScale"]>,
): (x: number) => number {
  if (points.length < 2) {
    return () => plotY(points[0].y, yScale);
  }

  const xs = points.map((point) => Math.log10(point.x));
  const ys = points.map((point) => plotY(point.y, yScale));
  const xMean = xs.reduce((sum, value) => sum + value, 0) / xs.length;
  const yMean = ys.reduce((sum, value) => sum + value, 0) / ys.length;
  const covariance = xs.reduce((sum, x, index) => sum + (x - xMean) * (ys[index] - yMean), 0);
  const variance = xs.reduce((sum, x) => sum + (x - xMean) ** 2, 0);
  const slope = variance === 0 ? 0 : covariance / variance;
  const intercept = yMean - slope * xMean;
  return (x) => intercept + slope * x;
}

function plotY(value: number, scale: NonNullable<LineChartProps["yScale"]>): number {
  if (scale === "log") {
    if (value <= 0) throw new RangeError("Log-scale chart values must be positive.");
    return Math.log10(value);
  }
  return value;
}

function formatValue(
  value: number,
  format: NonNullable<LineChartProps["valueFormat"]>,
  precise = false,
): string {
  if (format === "percent") return `${(value * 100).toFixed(precise ? 2 : 0)}%`;
  if (format === "compact") {
    return new Intl.NumberFormat("en-US", {
      notation: "compact",
      maximumFractionDigits: precise ? 2 : 1,
    }).format(value);
  }
  return value.toFixed(precise ? 4 : 2);
}
