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

export type ComparisonSeries = {
  id: string;
  label: string;
  color: string;
  points: { name: string; x: number; y: number }[];
  fitPoints: { x: number; y: number }[];
};

type ChartPoint = {
  logX: number;
  x: number;
  [key: string]: number | string;
};

export function FamilyComparisonChart({
  label,
  series,
  xFormat,
  xLabel,
}: {
  label: string;
  series: ComparisonSeries[];
  xFormat: "flops" | "currency";
  xLabel: string;
}) {
  const chartPoints = mergePoints(series);
  const allY = series.flatMap((item) => [
    ...item.points.map((point) => point.y),
    ...item.fitPoints.map((point) => point.y),
  ]);
  const allX = series.flatMap((item) => [
    ...item.points.map((point) => point.x),
    ...item.fitPoints.map((point) => point.x),
  ]);
  const xTicks = niceLogTicks(Math.min(...allX), Math.max(...allX), xFormat);
  const yTicks = niceLinearTicks(Math.min(...allY), Math.max(...allY));

  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-x-5 gap-y-2 text-sm text-zinc-600">
        {series.map((item) => (
          <span className="flex items-center gap-2" key={item.id}>
            <span className="size-3 rounded-full" style={{ backgroundColor: item.color }} />
            {item.label}
          </span>
        ))}
      </div>
      <div aria-label={label} className="h-[420px] w-full" role="img">
        <ResponsiveContainer height="100%" width="100%">
          <ComposedChart
            data={chartPoints}
            margin={{ top: 12, right: 20, bottom: 18, left: 12 }}
          >
            <CartesianGrid stroke="#e4e4e7" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="logX"
              domain={
                xFormat === "currency"
                  ? [xTicks[0], xTicks.at(-1)!]
                  : ["dataMin", "dataMax"]
              }
              label={{ value: xLabel, position: "insideBottom", offset: -10 }}
              ticks={xTicks}
              tickFormatter={(value: number) => formatX(10 ** value, xFormat)}
              type="number"
            />
            <YAxis
              domain={[yTicks[0], yTicks.at(-1)!]}
              label={{ value: "Loss", angle: -90, position: "insideLeft" }}
              ticks={yTicks}
              tickFormatter={(value: number) => value.toFixed(1)}
              width={68}
            />
            <Tooltip
              cursor={{ stroke: "#a1a1aa", strokeDasharray: "3 3" }}
              content={({ active, payload }) => {
                const entries = payload?.filter(
                  (entry) =>
                    typeof entry.dataKey === "string" &&
                    entry.dataKey.startsWith("value_") &&
                    entry.value !== undefined,
                );
                if (!active || !entries?.length) return null;
                const point = entries[0].payload as ChartPoint;
                return (
                  <div className="rounded-md border border-zinc-200 bg-white px-3 py-2 text-sm shadow-lg">
                    <div className="font-semibold text-zinc-950">
                      {String(point[`name_${entries[0].dataKey}`])}
                    </div>
                    <div className="mt-1 text-zinc-600">
                      Loss: {Number(entries[0].value).toFixed(4)}
                    </div>
                    <div className="mt-1 text-zinc-500">
                      {xLabel}: {formatX(point.x, xFormat, true)}
                    </div>
                  </div>
                );
              }}
            />
            {series.flatMap((item) =>
              item.fitPoints.slice(1).map((point, index) => (
                <ReferenceLine
                  ifOverflow="extendDomain"
                  key={`${item.id}-${point.x}`}
                  segment={[
                    {
                      x: Math.log10(item.fitPoints[index].x),
                      y: item.fitPoints[index].y,
                    },
                    { x: Math.log10(point.x), y: point.y },
                  ]}
                  stroke={item.color}
                  strokeOpacity={0.7}
                  strokeWidth={1.5}
                />
              )),
            )}
            {series.map((item) => (
              <Line
                activeDot={{ fill: item.color, r: 6, stroke: item.color }}
                connectNulls
                dataKey={`value_${item.id}`}
                dot={{ fill: item.color, r: 5, stroke: item.color }}
                isAnimationActive={false}
                key={item.id}
                stroke="none"
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function mergePoints(series: ComparisonSeries[]): ChartPoint[] {
  const points = new Map<number, ChartPoint>();
  for (const item of series) {
    for (const point of item.points) {
      const logX = Math.log10(point.x);
      points.set(logX, {
        ...points.get(logX),
        logX,
        x: point.x,
        [`name_value_${item.id}`]: `${item.label} ${point.name}`,
        [`value_${item.id}`]: point.y,
      });
    }
  }
  return [...points.values()].sort((left, right) => left.logX - right.logX);
}

function formatX(value: number, format: "flops" | "currency", precise = false) {
  if (format === "flops") return value.toExponential(precise ? 3 : 0).replace("e+", "e");
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: precise ? 3 : value < 1 ? 2 : 1,
  }).format(value);
}

function niceLinearTicks(min: number, max: number, targetCount = 6): number[] {
  const roughStep = (max - min) / (targetCount - 1);
  const magnitude = 10 ** Math.floor(Math.log10(roughStep));
  const normalized = roughStep / magnitude;
  const multiplier = [1, 2, 2.5, 5, 10].find((candidate) => candidate >= normalized) ?? 10;
  const step = multiplier * magnitude;
  const start = Math.floor(min / step) * step;
  const end = Math.ceil(max / step) * step;
  return Array.from(
    { length: Math.round((end - start) / step) + 1 },
    (_, index) => start + index * step,
  );
}

function niceLogTicks(
  min: number,
  max: number,
  format: "flops" | "currency",
): number[] {
  const candidates = [];
  const multipliers = format === "currency" ? [1, 3] : [1];
  for (
    let exponent = Math.floor(Math.log10(min)) - 1;
    exponent <= Math.ceil(Math.log10(max)) + 1;
    exponent += 1
  ) {
    for (const multiplier of multipliers) {
      candidates.push(multiplier * 10 ** exponent);
    }
  }
  const lower = candidates.filter((value) => value <= min).at(-1)!;
  const upper = candidates.find((value) => value >= max)!;
  if (format === "flops") {
    return candidates
      .filter((value) => value >= min && value <= max)
      .map(Math.log10);
  }
  return candidates
    .filter((value) => value >= lower && value <= upper)
    .map(Math.log10);
}
