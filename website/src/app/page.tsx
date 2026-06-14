import Link from "next/link";
import { MetricCard } from "@/components/metric-card";
import { Sparkline } from "@/components/sparkline";
import {
  formatCompactNumber,
  formatDecimal,
  formatModel,
  latestRun,
  modelFamilies,
  readBestRuns,
} from "@/data/best-runs";

export default function Home() {
  const families = modelFamilies.map((family) => {
    const runs = readBestRuns(family);
    return { family, runs, latest: latestRun(runs) };
  });

  const bestLatestLoss = Math.min(
    ...families
      .map(({ latest }) => latest?.lossUpper1sd ?? latest?.loss)
      .filter((loss): loss is number => loss !== undefined),
  );

  return (
    <main className="min-h-screen bg-zinc-50">
      <section className="border-b border-zinc-200 bg-white">
        <div className="mx-auto flex max-w-6xl flex-col gap-8 px-6 py-10">
          <div className="max-w-3xl">
            <p className="text-sm font-semibold uppercase tracking-[0.14em] text-blue-700">
              Chess Engine 4
            </p>
            <h1 className="mt-3 text-4xl font-semibold tracking-tight text-zinc-950">
              Scaling-law dashboard
            </h1>
            <p className="mt-4 text-lg leading-8 text-zinc-600">
              Public report site for lc0-compatible neural net experiments. The data comes from
              committed best-run TOML files in the Python repo.
            </p>
          </div>
          <div className="grid gap-4 md:grid-cols-3">
            <MetricCard label="Model families" value={String(families.length)} />
            <MetricCard
              label="Best current loss score"
              value={formatDecimal(bestLatestLoss)}
              detail="Lowest latest loss upper 1 SD"
            />
            <MetricCard label="Largest run" value="1e21" detail="Current best-run frontier" />
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-6xl gap-5 px-6 py-8">
        {families.map(({ family, runs, latest }) => (
          <Link
            key={family.id}
            href={`/families/${family.id}`}
            className="grid gap-5 rounded-lg border border-zinc-200 bg-white p-5 shadow-sm transition hover:border-blue-300 hover:shadow-md md:grid-cols-[1.2fr_1fr]"
          >
            <div>
              <div className="flex flex-wrap items-center gap-3">
                <h2 className="text-2xl font-semibold text-zinc-950">{family.name}</h2>
                {latest ? (
                  <span className="rounded-full bg-zinc-100 px-3 py-1 text-sm font-medium text-zinc-700">
                    latest {latest.budget}
                  </span>
                ) : null}
              </div>
              <p className="mt-3 max-w-2xl text-base leading-7 text-zinc-600">
                {family.description}
              </p>
              {latest ? (
                <dl className="mt-5 grid grid-cols-2 gap-4 text-sm md:grid-cols-4">
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
                    <dt className="text-zinc-500">Loss score</dt>
                    <dd className="mt-1 font-semibold text-zinc-950">
                      {formatDecimal(latest.lossUpper1sd ?? latest.loss)}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-zinc-500">Policy top-1</dt>
                    <dd className="mt-1 font-semibold text-zinc-950">
                      {formatDecimal(latest.policyTop1)}
                    </dd>
                  </div>
                </dl>
              ) : null}
            </div>
            <div className="rounded-md border border-zinc-200 bg-zinc-50 p-4">
              <div className="mb-2 text-sm font-medium text-zinc-500">Loss by compute</div>
              <Sparkline
                label={`${family.name} loss trend`}
                points={runs.map((run) => ({
                  x: run.compute,
                  y: run.lossUpper1sd ?? run.loss,
                }))}
              />
            </div>
          </Link>
        ))}
      </section>
    </main>
  );
}
