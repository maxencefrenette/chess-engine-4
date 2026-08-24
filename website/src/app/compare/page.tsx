import Link from "next/link";
import {
  FamilyComparisonChart,
  type ComparisonSeries,
} from "@/components/family-comparison-chart";
import { currentRuns, modelFamilies } from "@/data/best-runs";
import { trainingCost } from "@/data/format";

const FAMILY_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"];

export const metadata = {
  title: "Dense scaling curves",
};

export default function ComparePage() {
  const families = modelFamilies;
  const flopsSeries: ComparisonSeries[] = families.map((family, index) => ({
    id: family.id,
    label: family.name,
    color: FAMILY_COLORS[index % FAMILY_COLORS.length],
    points: currentRuns(family).map((run) => ({
      name: run.name,
      x: run.physicalFlops,
      y: run.loss,
    })),
    fitPoints: family.curves.loss.map((point) => ({
      x: point.physicalFlops,
      y: point.value,
    })),
  }));
  const costSeries: ComparisonSeries[] = families.map((family, index) => ({
    id: family.id,
    label: family.name,
    color: FAMILY_COLORS[index % FAMILY_COLORS.length],
    points: currentRuns(family).map((run) => ({
      name: run.name,
      x: trainingCost(run),
      y: run.loss,
    })),
    fitPoints: [],
  }));

  return (
    <main>
      <div className="site-shell py-10 sm:py-14">
        <header className="border-b border-rule pb-7">
          <Link href="/experiments" className="text-link text-sm font-medium">
            Experiments
          </Link>
          <h1 className="mt-3 text-4xl font-semibold tracking-[-0.04em] text-ink">
            Dense scaling curves
          </h1>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-body">
            Final loss for canonical dense runs. Dollar cost uses the
            recipe&apos;s selected GPU, measured steady-state throughput, and eight Modal CPU cores.
          </p>
        </header>

        <section className="mt-7 grid gap-5 xl:grid-cols-2">
          <ChartCard title="Loss by training FLOPs">
            <FamilyComparisonChart
              label="Loss by training FLOPs"
              series={flopsSeries}
              xFormat="flops"
              xLabel="Training FLOPs"
            />
          </ChartCard>
          <ChartCard title="Loss by training cost">
            <FamilyComparisonChart
              label="Loss by estimated training cost"
              series={costSeries}
              xFormat="currency"
              xLabel="Estimated cost"
            />
          </ChartCard>
        </section>
      </div>
    </main>
  );
}

function ChartCard({ children, title }: { children: React.ReactNode; title: string }) {
  return (
    <div className="editorial-card p-5">
      <h2 className="text-lg font-semibold text-ink">{title}</h2>
      <div className="mt-4">{children}</div>
    </div>
  );
}
