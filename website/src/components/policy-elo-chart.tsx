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
import type { PolicyEloData, PolicyEloPoint } from "@/data/best-runs";

type PlotPoint = PolicyEloPoint & { logCost: number };

export function PolicyEloChart({ data, compact = false }: { data: PolicyEloData; compact?: boolean }) {
  const dense = plotPoints(data.points.filter((point) => point.family === "dense"));
  const trend = data.trend.points.map((point) => ({ ...point, logCost: Math.log10(point.cost) }));

  return (
    <div>
      <div className="mb-4 flex flex-wrap gap-x-5 gap-y-2 font-mono text-[11px] uppercase tracking-[0.12em] text-muted">
        <Legend color="#2457d6" label="Dense" />
        <span className="flex items-center gap-2">
          <span className="h-px w-5 border-t border-dashed border-ink" /> Current trend
        </span>
        <span className="flex items-center gap-2">
          <span className="h-px w-5 bg-cobalt" /> BT4 target
        </span>
      </div>
      <div
        aria-label="Policy Elo by estimated training cost"
        className={compact ? "h-[330px] w-full" : "h-[480px] w-full"}
        role="img"
      >
        <ResponsiveContainer height="100%" width="100%">
          <ComposedChart margin={{ top: 12, right: 18, bottom: 20, left: 8 }}>
            <CartesianGrid stroke="#d8d2c5" strokeDasharray="2 5" />
            <XAxis
              dataKey="logCost"
              domain={[-2.2, 5]}
              label={{ value: "Estimated training cost", position: "insideBottom", offset: -12 }}
              ticks={[-2, -1, 0, 1, 2, 3, 4, 5]}
              tickFormatter={(value: number) => formatCost(10 ** value)}
              type="number"
            />
            <YAxis
              dataKey="elo"
              domain={[-1500, 200]}
              label={{ value: "Policy Elo", angle: -90, position: "insideLeft" }}
              ticks={[-1500, -1250, -1000, -750, -500, -250, 0]}
              width={62}
            />
            <Tooltip content={<PolicyTooltip />} cursor={{ stroke: "#8e887c", strokeDasharray: "2 4" }} />
            <ReferenceLine
              label={{ value: "BT4", fill: "#2457d6", fontSize: 12, position: "insideTopLeft" }}
              stroke="#2457d6"
              strokeWidth={1.5}
              y={data.target.elo}
            />
            <Line
              data={trend}
              dataKey="elo"
              dot={false}
              isAnimationActive={false}
              stroke="#24231f"
              strokeDasharray="6 6"
              strokeOpacity={0.72}
              strokeWidth={1.5}
              type="linear"
              xAxisId={0}
            />
            <Scatter data={dense} fill="#2457d6" isAnimationActive={false} name="Dense" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function plotPoints(points: PolicyEloPoint[]): PlotPoint[] {
  return points.map((point) => ({ ...point, logCost: Math.log10(point.cost) }));
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-2">
      <span className="size-2.5 rounded-full" style={{ backgroundColor: color }} />
      {label}
    </span>
  );
}

function PolicyTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: PlotPoint }>;
}) {
  const point = payload?.find((entry) => "family" in entry.payload)?.payload;
  if (!active || !point) return null;
  return (
    <div className="border border-rule bg-paper px-3 py-2 text-sm shadow-lg">
      <div className="font-semibold text-ink">Dense {point.name}</div>
      <div className="mt-1 text-body">{point.elo.toFixed(0)} policy Elo</div>
      <div className="font-mono text-xs text-muted">{formatCost(point.cost)} estimated training</div>
    </div>
  );
}

function formatCost(cost: number): string {
  if (cost < 0.1) return `$${cost.toFixed(2)}`;
  if (cost < 1_000) return `$${new Intl.NumberFormat("en-US", { maximumFractionDigits: cost < 10 ? 1 : 0 }).format(cost)}`;
  return `$${new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 0 }).format(cost)}`;
}
