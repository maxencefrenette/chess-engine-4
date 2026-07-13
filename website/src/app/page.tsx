import Link from "next/link";
import { Sparkline } from "@/components/sparkline";
import {
  formatCompactNumber,
  formatDecimal,
  formatModel,
  formatPercent,
} from "@/data/format";
import {
  latestRun,
  modelFamilies,
  readBestRuns,
  readScalingFamily,
} from "@/data/best-runs";

export default function Home() {
  const families = modelFamilies.map((family) => {
    const runs = readBestRuns(family);
    const scaling = readScalingFamily(family);
    return { family, runs, staleRuns: scaling.staleObserved, latest: latestRun(runs) };
  });

  const bestLatestLoss = Math.min(
    ...families
      .map(({ latest }) => latest?.loss)
      .filter((loss): loss is number => loss !== undefined),
  );
  const largestRun = families
    .flatMap(({ runs }) => runs)
    .reduce((largest, run) =>
      run.physicalFlops > largest.physicalFlops ? run : largest,
    ).name;

  return (
    <main className="min-h-screen bg-zinc-50">
      <section className="border-b border-zinc-200 bg-white">
        <div className="mx-auto flex max-w-[1500px] items-end justify-between gap-8 px-8 py-6">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.14em] text-blue-700">
              Chess Engine 4
            </p>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight text-zinc-950">
              Scaling laws
            </h1>
          </div>
          <div className="flex gap-8 text-right tabular-nums">
            <div>
              <div className="text-xs font-medium uppercase text-zinc-500">Families</div>
              <div className="mt-1 text-xl font-semibold text-zinc-950">{families.length}</div>
            </div>
            <div>
              <div className="text-xs font-medium uppercase text-zinc-500">Best frontier loss</div>
              <div className="mt-1 text-xl font-semibold text-zinc-950">
                {formatDecimal(bestLatestLoss)}
              </div>
            </div>
            <div>
              <div className="text-xs font-medium uppercase text-zinc-500">Largest run</div>
              <div className="mt-1 text-xl font-semibold text-zinc-950">{largestRun}</div>
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-[1500px] grid-cols-1 gap-5 px-8 py-6">
        {families.map(({ family, runs, staleRuns, latest }) => (
          <Link
            key={family.id}
            href={`/families/${family.id}`}
            className="grid gap-5 rounded-lg border border-zinc-200 bg-white p-5 shadow-sm transition hover:border-blue-300 hover:shadow-md xl:grid-cols-[1.05fr_1fr]"
          >
            <div>
              <div className="flex flex-wrap items-center gap-3">
                <h2 className="text-2xl font-semibold text-zinc-950">{family.name}</h2>
                {latest ? (
                  <span className="rounded-full bg-zinc-100 px-3 py-1 text-sm font-medium text-zinc-700">
                    latest {latest.name}
                  </span>
                ) : null}
              </div>
              <p className="mt-2 max-w-2xl text-sm leading-6 text-zinc-600">
                {family.description}
              </p>
              {latest ? (
                <dl className="mt-5 grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <dt className="text-zinc-500">Model</dt>
                    <dd className="mt-1 font-semibold text-zinc-950">{formatModel(latest)}</dd>
                  </div>
                  <div>
                    <dt className="text-zinc-500">Params</dt>
                    <dd className="mt-1 font-semibold text-zinc-950">
                      {formatCompactNumber(latest.params)}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-zinc-500">Loss</dt>
                    <dd className="mt-1 font-semibold text-zinc-950">
                      {formatDecimal(latest.loss)}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-zinc-500">Policy top-1</dt>
                    <dd className="mt-1 font-semibold text-zinc-950">
                      {formatPercent(latest.policyTop1)}
                    </dd>
                  </div>
                </dl>
              ) : null}
            </div>
            <div className="rounded-md border border-zinc-200 bg-zinc-50 p-4">
              <div className="mb-2 text-sm font-medium text-zinc-500">Loss by training FLOPs</div>
              <Sparkline
                label={`${family.name} loss trend`}
                points={runs.map((run) => ({
                  x: run.physicalFlops,
                  y: run.loss,
                }))}
                stalePoints={staleRuns.map((run) => ({
                  x: run.physicalFlops,
                  y: run.loss,
                }))}
              />
            </div>
          </Link>
        ))}
      </section>
    </main>
  );
}
