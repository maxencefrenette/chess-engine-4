import Link from "next/link";
import {
  FamilyComparisonChart,
  type ComparisonSeries,
} from "@/components/family-comparison-chart";
import { modelFamilies, readScalingFamily } from "@/data/best-runs";
import { trainingCost } from "@/data/format";

const FAMILY_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"];

export const metadata = {
  title: "Model family comparison",
};

export default function ComparePage() {
  const families = modelFamilies.map(readScalingFamily);
  const flopsSeries: ComparisonSeries[] = families.map((family, index) => ({
    id: family.id,
    label: family.name,
    color: FAMILY_COLORS[index % FAMILY_COLORS.length],
    points: family.observed.map((run) => ({
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
    points: family.observed.map((run) => ({
      name: run.name,
      x: trainingCost(run),
      y: run.loss,
    })),
    fitPoints: [],
  }));

  return (
    <main className="min-h-screen bg-zinc-50">
      <div className="mx-auto max-w-[1500px] px-8 py-6">
        <header className="border-b border-zinc-200 pb-5">
          <Link href="/" className="text-sm font-medium text-blue-700 hover:text-blue-900">
            Families
          </Link>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight text-zinc-950">
            Family comparison
          </h1>
          <p className="mt-2 text-sm text-zinc-600">
            Final loss for canonical runs across model families. Dollar cost uses each
            recipe&apos;s selected GPU, measured steady-state throughput, and eight Modal CPU cores.
          </p>
        </header>

        <section className="mt-5 grid grid-cols-2 gap-5">
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
    <div className="rounded-lg border border-zinc-200 bg-white p-5 shadow-sm">
      <h2 className="text-lg font-semibold text-zinc-950">{title}</h2>
      <div className="mt-4">{children}</div>
    </div>
  );
}
