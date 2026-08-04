import Link from "next/link";
import { notFound } from "next/navigation";
import { FamilyDashboard } from "@/components/family-dashboard";
import { MetricCard } from "@/components/metric-card";
import { RunsTable } from "@/components/runs-table";
import {
  formatCompactNumber,
  formatDecimal,
  formatPercent,
} from "@/data/format";
import {
  getFamily,
  latestRun,
  modelFamilies,
  readBestRuns,
  readScalingFamily,
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
  const scalingFamily = readScalingFamily(family);
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
            <p className="mt-1 text-sm text-zinc-500">
              All runs use {scalingFamily.trainingRatio}x Chinchilla to reduce experiment cost.
            </p>
          </div>
          {latest ? (
            <div className="text-right">
              <div className="text-xs font-medium uppercase text-zinc-500">
                Current curve
              </div>
              <div className="mt-1 text-xl font-semibold text-zinc-950">{latest.name}</div>
            </div>
          ) : null}
        </header>

        {latest ? (
          <section className="mt-5 grid grid-cols-4 gap-4">
            <MetricCard
              label="Training FLOPs"
              value={`${latest.physicalFlops.toExponential(0).replace("e+", "e")} FLOPs`}
            />
            <MetricCard label="Parameters" value={formatCompactNumber(latest.params)} />
            <MetricCard
              label="Loss"
              value={formatDecimal(latest.loss)}
            />
            <MetricCard label="Policy top-1" value={formatPercent(latest.policyTop1)} />
          </section>
        ) : null}

        <FamilyDashboard family={scalingFamily} />

        <section className="mt-5 overflow-hidden rounded-lg border border-zinc-200 bg-white shadow-sm">
          <div className="border-b border-zinc-200 px-5 py-4">
            <h2 className="text-lg font-semibold text-zinc-950">
              Selected runs
            </h2>
          </div>
          <RunsTable runs={runs} />
        </section>
      </div>
    </main>
  );
}
