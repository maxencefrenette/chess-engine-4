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
  currentRuns,
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

  const runs = currentRuns(family);
  const latest = latestRun(runs);

  return (
    <main>
      <div className="site-shell py-10 sm:py-14">
        <header className="flex flex-col gap-8 border-b border-rule pb-7 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <nav className="flex gap-5 text-sm font-medium text-cobalt">
              <Link href="/experiments" className="hover:text-ink">
                Experiments
              </Link>
              <Link href="/compare" className="hover:text-ink">
                Compare
              </Link>
            </nav>
            <h1 className="mt-3 text-4xl font-semibold tracking-[-0.04em] text-ink">
              {family.name}
            </h1>
            <p className="mt-2 text-sm text-body">{family.description}</p>
            <p className="mt-1 text-sm text-muted">
              All runs use {family.trainingRatio}x Chinchilla to reduce experiment cost.
            </p>
            <p className="mt-1 text-sm text-zinc-500">
              Loss projections use the coupled Skaling law over model parameters and training samples.
            </p>
            {family.id === "dense" ? (
              <p className="mt-1 text-sm font-medium text-cobalt">
                RTX PRO 6000 reduces estimated cost by 37-48% through d256; d512 and larger remain on B200.
              </p>
            ) : null}
          </div>
          {latest ? (
            <div className="text-right">
              <div className="text-xs font-medium uppercase text-muted">
                Current curve
              </div>
              <div className="mt-1 text-xl font-semibold text-ink">{latest.name}</div>
            </div>
          ) : null}
        </header>

        {latest ? (
          <section className="mt-7 grid grid-cols-2 gap-4 lg:grid-cols-4">
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

        <FamilyDashboard family={family} />

        <section className="editorial-card mt-5 overflow-hidden">
          <div className="border-b border-rule px-5 py-4">
            <h2 className="text-lg font-semibold text-ink">
              Selected runs
            </h2>
          </div>
          <RunsTable
            runs={[...family.runs].sort(
              (left, right) => left.physicalFlops - right.physicalFlops,
            )}
          />
        </section>
      </div>
    </main>
  );
}
