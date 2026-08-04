"use client";

import type { BestRun } from "@/data/best-runs";
import {
  formatCompactNumber,
  formatB200Cost,
  formatDecimal,
  formatDuration,
  formatPercent,
} from "@/data/format";

export function RunsTable({ runs }: { runs: BestRun[] }) {
  return (
    <div className="max-h-[520px] overflow-auto">
      <table className="w-full min-w-[1080px] text-sm tabular-nums">
        <thead className="sticky top-0 z-10 bg-zinc-100 text-zinc-500 shadow-[0_1px_0_#e4e4e7]">
          <tr>
            <Th>Run</Th>
            <Th numeric>Depth</Th>
            <Th numeric>Batch</Th>
            <Th numeric>LR</Th>
            <Th numeric>Params</Th>
            <Th numeric>Samples</Th>
            <Th numeric>Loss</Th>
            <Th numeric>Policy top-1</Th>
            <Th numeric>Runtime</Th>
            <Th numeric>
              <span title="Estimated from recorded runtime at $6.25 per Modal B200-hour">
                Cost
              </span>
            </Th>
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-100">
          {runs.map((run) => (
            <tr
              key={`${run.name}-${run.physicalFlops}`}
              className={`cursor-pointer transition-colors hover:bg-blue-50 ${run.stale ? "bg-zinc-50/60 text-zinc-400" : "text-zinc-700"}`}
              onClick={() => window.open(run.wandbUrl, "_blank", "noopener,noreferrer")}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  window.open(run.wandbUrl, "_blank", "noopener,noreferrer");
                }
              }}
              role="link"
              tabIndex={0}
              title={`Open ${run.runName} in W&B`}
            >
              <Td strong={!run.stale}>
                <a
                  href={run.wandbUrl}
                  onClick={(event) => event.stopPropagation()}
                  rel="noreferrer"
                  target="_blank"
                >
                  {run.name}
                </a>
                {run.stale ? (
                  <span className="ml-2 text-xs font-medium uppercase text-zinc-400">
                    stale
                  </span>
                ) : null}
              </Td>
              <Td numeric>{run.depth}</Td>
              <Td numeric>{formatCompactNumber(run.batchSize)}</Td>
              <Td numeric>{run.lr.toExponential(1)}</Td>
              <Td numeric>{formatCompactNumber(run.params)}</Td>
              <Td numeric>{formatCompactNumber(run.samplesSeen)}</Td>
              <Td numeric>{formatDecimal(run.loss)}</Td>
              <Td numeric>{formatPercent(run.policyTop1)}</Td>
              <Td numeric>{formatDuration(run.runtimeSec)}</Td>
              <Td numeric>{formatB200Cost(run.runtimeSec)}</Td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Th({ children, numeric = false }: { children: React.ReactNode; numeric?: boolean }) {
  return (
    <th className={`whitespace-nowrap px-4 py-3 font-medium ${numeric ? "text-right" : "text-left"}`}>
      {children}
    </th>
  );
}

function Td({
  children,
  numeric = false,
  strong = false,
}: {
  children: React.ReactNode;
  numeric?: boolean;
  strong?: boolean;
}) {
  return (
    <td
      className={`whitespace-nowrap px-4 py-3 ${numeric ? "text-right" : "text-left"} ${strong ? "font-semibold text-zinc-950" : ""}`}
    >
      {children}
    </td>
  );
}
