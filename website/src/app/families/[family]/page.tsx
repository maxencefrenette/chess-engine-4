import Link from "next/link";
import { notFound } from "next/navigation";
import { LineChart } from "@/components/line-chart";
import { MetricCard } from "@/components/metric-card";
import {
  formatCompactNumber,
  formatDecimal,
  formatModel,
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
      <div className="mx-auto max-w-6xl px-6 py-8">
        <Link href="/" className="text-sm font-medium text-blue-700 hover:text-blue-900">
          Back to families
        </Link>

        <header className="mt-6 rounded-lg border border-zinc-200 bg-white p-6 shadow-sm">
          <p className="text-sm font-semibold uppercase tracking-[0.14em] text-blue-700">
            Model family
          </p>
          <h1 className="mt-3 text-4xl font-semibold tracking-tight text-zinc-950">
            {family.name}
          </h1>
          <p className="mt-4 max-w-3xl text-lg leading-8 text-zinc-600">
            {family.description}
          </p>
        </header>

        {latest ? (
          <section className="mt-5 grid gap-4 md:grid-cols-4">
            <MetricCard label="Latest model" value={formatModel(latest)} detail={latest.budget} />
            <MetricCard label="Parameters" value={formatCompactNumber(latest.params)} />
            <MetricCard
              label="Loss score"
              value={formatDecimal(latest.lossUpper1sd ?? latest.loss)}
              detail="Loss upper 1 SD when available"
            />
            <MetricCard label="Policy top-1" value={formatDecimal(latest.policyTop1)} />
          </section>
        ) : null}

        <section className="mt-5 grid gap-5 lg:grid-cols-2">
          <ChartCard
            title="Loss score"
            yLabel="Loss upper 1 SD"
            points={runs.map((run) => ({ x: run.compute, y: run.lossUpper1sd ?? run.loss }))}
          />
          <ChartCard
            title="Policy top-1"
            yLabel="Top-1 accuracy"
            points={runs.map((run) => ({ x: run.compute, y: run.policyTop1 }))}
            stroke="#16a34a"
          />
          <ChartCard
            title="Parameters"
            yLabel="Total parameters"
            points={runs.map((run) => ({ x: run.compute, y: run.params }))}
            stroke="#9333ea"
          />
          <ChartCard
            title="Samples seen"
            yLabel="Training samples"
            points={runs.map((run) => ({ x: run.compute, y: run.samplesSeen }))}
            stroke="#ea580c"
          />
        </section>

        <section className="mt-5 overflow-hidden rounded-lg border border-zinc-200 bg-white shadow-sm">
          <div className="border-b border-zinc-200 px-5 py-4">
            <h2 className="text-lg font-semibold text-zinc-950">Best observed runs</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[880px] text-left text-sm">
              <thead className="bg-zinc-50 text-zinc-500">
                <tr>
                  <Th>Budget</Th>
                  <Th>Model</Th>
                  <Th>Batch</Th>
                  <Th>LR</Th>
                  <Th>Params</Th>
                  <Th>Samples</Th>
                  <Th>Loss</Th>
                  <Th>Loss score</Th>
                  <Th>Policy top-1</Th>
                  <Th>W&B</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100">
                {runs.map((run) => (
                  <tr key={run.budget} className="text-zinc-700">
                    <Td>{run.budget}</Td>
                    <Td>{formatModel(run)}</Td>
                    <Td>{formatCompactNumber(run.batchSize)}</Td>
                    <Td>{run.lr.toPrecision(2)}</Td>
                    <Td>{formatCompactNumber(run.params)}</Td>
                    <Td>{formatCompactNumber(run.samplesSeen)}</Td>
                    <Td>{formatDecimal(run.loss)}</Td>
                    <Td>{formatDecimal(run.lossUpper1sd ?? run.loss)}</Td>
                    <Td>{formatDecimal(run.policyTop1)}</Td>
                    <Td>
                      <a
                        href={run.wandbUrl}
                        className="font-medium text-blue-700 hover:text-blue-900"
                      >
                        run
                      </a>
                    </Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
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
}: {
  title: string;
  yLabel: string;
  points: { x: number; y: number }[];
  stroke?: string;
}) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-5 shadow-sm">
      <h2 className="text-lg font-semibold text-zinc-950">{title}</h2>
      <div className="mt-4">
        <LineChart
          label={title}
          points={points}
          stroke={stroke}
          xLabel="Compute budget"
          yLabel={yLabel}
        />
      </div>
    </div>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return <th className="px-4 py-3 font-medium">{children}</th>;
}

function Td({ children }: { children: React.ReactNode }) {
  return <td className="px-4 py-3">{children}</td>;
}
