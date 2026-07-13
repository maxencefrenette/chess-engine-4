"use client";

import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type LineChartPoint = {
  x: number;
  y: number;
  name: string;
};

type LineChartProps = {
  points: LineChartPoint[];
  extrapolatedPoints: LineChartPoint[];
  fitPoints: { x: number; y: number }[];
  label: string;
  xLabel: string;
  yLabel: string;
  stroke?: string;
  valueFormat?: "decimal" | "percent" | "compact" | "scientific";
  yScale?: "linear" | "log";
};

type ChartPoint = {
  name: string;
  extrapolated?: boolean;
  fit?: number;
  logX: number;
  plotY?: number;
  x: number;
  y?: number;
};

export function LineChart({
  points,
  extrapolatedPoints,
  fitPoints,
  label,
  xLabel,
  yLabel,
  stroke = "#2563eb",
  valueFormat = "decimal",
  yScale = "linear",
}: LineChartProps) {
  if (points.length === 0) {
    return null;
  }

  const observedChartPoints: ChartPoint[] = points.map((point) => ({
    ...point,
    logX: Math.log10(point.x),
    plotY: plotY(point.y, yScale),
  }));
  const extrapolatedChartPoints: ChartPoint[] = extrapolatedPoints.map((point) => ({
    ...point,
    extrapolated: true,
    logX: Math.log10(point.x),
    plotY: plotY(point.y, yScale),
  }));
  const curveChartPoints: ChartPoint[] = fitPoints.map((point) => ({
    name: "Fit",
    x: point.x,
    y: point.y,
    logX: Math.log10(point.x),
    fit: plotY(point.y, yScale),
  }));
  const chartPoints = mergeChartPoints([...observedChartPoints, ...extrapolatedChartPoints]);
  const yValues = [...curveChartPoints, ...chartPoints].flatMap((point) =>
    [point.plotY, point.fit].filter(
      (value): value is number => value !== undefined,
    ),
  );
  const yTicks = niceTicks(yValues, yScale);

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
            domain={[yTicks[0], yTicks.at(-1)!]}
            label={{ value: yLabel, angle: -90, position: "insideLeft" }}
            ticks={yTicks}
            tickFormatter={(value: number) =>
              formatValue(yScale === "log" ? 10 ** value : value, valueFormat)
            }
            width={68}
          />
          <Tooltip
            cursor={{ stroke: "#a1a1aa", strokeDasharray: "3 3" }}
            content={({ active, payload }) => {
              const point = payload?.find(
                (entry) => entry.dataKey === "plotY" && entry.value !== undefined,
              )?.payload as ChartPoint | undefined;
              if (!active || !point || point.y === undefined) return null;
              return (
                <div className="rounded-md border border-zinc-200 bg-white px-3 py-2 text-sm shadow-lg">
                  <div className="font-semibold text-zinc-950">{point.name}</div>
                  <div className="mt-1 text-zinc-600">
                    {yLabel}: {formatValue(point.y, valueFormat, true)}
                  </div>
                  <div className="mt-1 text-zinc-500">
                    {xLabel}: {point.x.toExponential(3)}
                  </div>
                </div>
              );
            }}
          />
          {curveChartPoints.slice(1).map((point, index) => (
            <ReferenceLine
              key={point.logX}
              ifOverflow="extendDomain"
              segment={[
                { x: curveChartPoints[index].logX, y: curveChartPoints[index].fit },
                { x: point.logX, y: point.fit },
              ]}
              stroke={stroke}
              strokeDasharray="5 5"
              strokeOpacity={0.65}
              strokeWidth={1.5}
            />
          ))}
          <Line
            activeDot={<ChartDot active stroke={stroke} />}
            connectNulls={false}
            dataKey="plotY"
            dot={<ChartDot stroke={stroke} />}
            isAnimationActive={false}
            stroke="none"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function ChartDot({
  active = false,
  cx = 0,
  cy = 0,
  payload,
  stroke,
}: {
  active?: boolean;
  cx?: number;
  cy?: number;
  payload?: ChartPoint;
  stroke: string;
}) {
  const extrapolated = payload?.extrapolated === true;
  return (
    <circle
      cx={cx}
      cy={cy}
      fill={extrapolated ? "white" : stroke}
      r={(extrapolated ? 5.5 : 5) + (active ? 1 : 0)}
      stroke={stroke}
      strokeWidth={extrapolated ? 2 : 1}
    />
  );
}

function mergeChartPoints(points: ChartPoint[]): ChartPoint[] {
  const merged = new Map<number, ChartPoint>();
  for (const point of points) {
    merged.set(point.logX, { ...merged.get(point.logX), ...point });
  }
  return [...merged.values()].sort((left, right) => left.logX - right.logX);
}

function plotY(value: number, scale: NonNullable<LineChartProps["yScale"]>): number {
  if (scale === "log") {
    if (value <= 0) throw new RangeError("Log-scale chart values must be positive.");
    return Math.log10(value);
  }
  return value;
}

function niceTicks(
  values: number[],
  scale: NonNullable<LineChartProps["yScale"]>,
): number[] {
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (scale === "linear") return niceLinearTicks(min, max);

  if (max - min >= 1.5) {
    return Array.from(
      { length: Math.ceil(max) - Math.floor(min) + 1 },
      (_, index) => Math.floor(min) + index,
    );
  }
  return niceLinearTicks(10 ** min, 10 ** max)
    .filter((value) => value > 0)
    .map(Math.log10);
}

function niceLinearTicks(min: number, max: number, targetCount = 6): number[] {
  if (min === max) {
    const padding = Math.abs(min) * 0.1 || 1;
    return [min - padding, min, min + padding];
  }
  const roughStep = (max - min) / (targetCount - 1);
  const magnitude = 10 ** Math.floor(Math.log10(roughStep));
  const normalized = roughStep / magnitude;
  const multiplier = [1, 2, 2.5, 5, 10].find((candidate) => candidate >= normalized) ?? 10;
  const step = multiplier * magnitude;
  const start = Math.floor(min / step) * step;
  const end = Math.ceil(max / step) * step;
  const count = Math.round((end - start) / step) + 1;
  return Array.from({ length: count }, (_, index) => start + index * step);
}

function formatValue(
  value: number,
  format: NonNullable<LineChartProps["valueFormat"]>,
  precise = false,
): string {
  if (format === "percent") return `${(value * 100).toFixed(precise ? 2 : 0)}%`;
  if (format === "scientific") return value.toExponential(precise ? 2 : 1);
  if (format === "compact") {
    return new Intl.NumberFormat("en-US", {
      notation: "compact",
      maximumFractionDigits: precise ? 2 : 1,
    }).format(value);
  }
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: precise ? 4 : 2,
  }).format(value);
}
