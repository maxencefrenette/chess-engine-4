import Link from "next/link";
import { notFound } from "next/navigation";
import { LineChart } from "@/components/line-chart";
import { MetricCard } from "@/components/metric-card";
import { RunsTable } from "@/components/runs-table";
import {
  formatCompactNumber,
  formatDecimal,
  formatModel,
  formatPercent,
} from "@/data/format";
import {
  getFamily,
  latestRun,
  modelFamilies,
  readBestRuns,
} from "@/data/best-runs";

type FamilyPageProps = {
  params: Promise<{ family: string }>;
};

export function generateStaticParams() {
  return modelFamilies.map((family) => ({ family: family.id }));
}

export async function generateMetadata({ params }: FamilyPageProps) {
  const { family: familyId } = await params;
  const family = getFamily(familyId);
  return {
    title: family ? `${family.name} scaling laws` : "Model family",
  };
}

export default async function FamilyPage({ params }: FamilyPageProps) {
  const { family: familyId } = await params;
  const family = getFamily(familyId);
  if (!family) {
    notFound();
  }

  const runs = readBestRuns(family);
  const latest = latestRun(runs);

  return (
    <main className="min-h-screen bg-zinc-50">
      <div className="mx-auto max-w-[1500px] px-8 py-6">
        <header className="flex items-end justify-between gap-8 border-b border-zinc-200 pb-5">
          <div>
            <Link href="/" className="text-sm font-medium text-blue-700 hover:text-blue-900">
              Families
            </Link>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight text-zinc-950">
              {family.name}
            </h1>
            <p className="mt-2 text-sm text-zinc-600">{family.description}</p>
          </div>
          {latest ? (
            <div className="text-right">
              <div className="text-xs font-medium uppercase text-zinc-500">Current frontier</div>
              <div className="mt-1 text-xl font-semibold text-zinc-950">{latest.budget}</div>
            </div>
          ) : null}
        </header>

        {latest ? (
          <section className="mt-5 grid grid-cols-4 gap-4">
            <MetricCard label="Latest model" value={formatModel(latest)} detail={latest.budget} />
            <MetricCard label="Parameters" value={formatCompactNumber(latest.params)} />
            <MetricCard
              label="Loss score"
              value={formatDecimal(latest.lossUpper1sd ?? latest.loss)}
              detail="Loss upper 1 SD when available"
            />
            <MetricCard label="Policy top-1" value={formatPercent(latest.policyTop1)} />
          </section>
        ) : null}

        <section className="mt-5 grid gap-5 lg:grid-cols-2">
          <ChartCard
            title="Loss score"
            yLabel="Loss upper 1 SD"
            points={runs.map((run) => ({
              x: run.compute,
              y: run.lossUpper1sd ?? run.loss,
              budget: run.budget,
            }))}
          />
          <ChartCard
            title="Policy top-1"
            yLabel="Top-1 accuracy"
            points={runs.map((run) => ({ x: run.compute, y: run.policyTop1, budget: run.budget }))}
            stroke="#16a34a"
            valueFormat="percent"
          />
          <ChartCard
            title="Parameters"
            yLabel="Total parameters"
            points={runs.map((run) => ({ x: run.compute, y: run.params, budget: run.budget }))}
            stroke="#9333ea"
            valueFormat="compact"
            yScale="log"
          />
          <ChartCard
            title="Samples seen"
            yLabel="Training samples"
            points={runs.map((run) => ({
              x: run.compute,
              y: run.samplesSeen,
              budget: run.budget,
            }))}
            stroke="#ea580c"
            valueFormat="compact"
            yScale="log"
          />
        </section>

        <section className="mt-5 overflow-hidden rounded-lg border border-zinc-200 bg-white shadow-sm">
          <div className="border-b border-zinc-200 px-5 py-4">
            <h2 className="text-lg font-semibold text-zinc-950">Best observed runs</h2>
          </div>
          <RunsTable runs={runs} />
        </section>
      </div>
    </main>
  );
}

function ChartCard({
  title,
  yLabel,
  points,
  stroke,
  valueFormat,
  yScale,
}: {
  title: string;
  yLabel: string;
  points: { x: number; y: number; budget: string }[];
  stroke?: string;
  valueFormat?: "decimal" | "percent" | "compact";
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
          valueFormat={valueFormat}
          yScale={yScale}
          xLabel="Compute budget"
          yLabel={yLabel}
        />
      </div>
    </div>
  );
}
