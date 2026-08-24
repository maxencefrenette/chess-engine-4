import Link from "next/link";
import { PolicyEloChart } from "@/components/policy-elo-chart";
import { policyElo, trainingDataset } from "@/data/best-runs";

const pathways = [
  {
    number: "01",
    title: "How it works",
    body: "Follow one LCZero record through conversion, training, export, and evaluation inside lc0.",
    href: "/how-it-works",
  },
  {
    number: "02",
    title: "Architecture",
    body: "See how a chess position becomes one learned state inside the final dense network.",
    href: "/architecture",
  },
  {
    number: "03",
    title: "Experiments",
    body: "Read the findings first, then inspect every canonical run and scaling fit.",
    href: "/experiments",
  },
];

export default function Home() {
  const estimatedTarget = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(policyElo.trend.estimatedTargetCost);
  const largestMeasuredCost = Math.max(...policyElo.points.map((point) => point.cost));

  return (
    <main>
      <section className="border-b border-rule py-10 sm:py-16 lg:py-20">
        <div className="site-shell grid items-end gap-10 lg:grid-cols-[0.78fr_1.22fr]">
          <div>
            <p className="eyebrow">Neural chess, end to end</p>
            <h1 className="display-title mt-5">A cost-efficient chess neural network that works in lc0.</h1>
            <p className="lead mt-7">
              Chess Engine 4 is a ground-up training system for LCZero-compatible networks:
              billions of supervised positions, measured scaling laws, dense models, custom
              kernels, and a native inference path.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link className="bg-ink px-5 py-3 text-sm font-semibold text-paper hover:bg-cobalt" href="/how-it-works">
                Follow the system
              </Link>
              <a
                className="border border-ink px-5 py-3 text-sm font-semibold text-ink hover:border-cobalt hover:text-cobalt"
                href="https://github.com/maxencefrenette/chess-engine-4"
                rel="noreferrer"
                target="_blank"
              >
                View source ↗
              </a>
            </div>
          </div>

          <div className="editorial-card p-4 sm:p-6">
            <div className="flex flex-col gap-2 border-b border-rule pb-4 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <p className="eyebrow">The current frontier</p>
                <h2 className="mt-2 text-xl font-semibold tracking-tight">Policy strength per training dollar</h2>
              </div>
            </div>
            <div className="pt-5">
              <PolicyEloChart compact data={policyElo} />
            </div>
            <div className="grid gap-4 border-t border-rule pt-4 text-sm sm:grid-cols-[1fr_auto] sm:items-start">
              <p className="leading-6 text-body">
                The fitted trend reaches BT4 near <strong className="text-ink">{estimatedTarget}</strong>.
                That is a long extrapolation, not a forecast—the gap is the point.
              </p>
              <Link className="text-link whitespace-nowrap font-semibold" href="/experiments/policy-elo">
                Read the method →
              </Link>
            </div>
          </div>
        </div>
      </section>

      <section className="site-shell py-16 sm:py-24">
        <div className="grid gap-8 lg:grid-cols-[0.7fr_1.3fr]">
          <div>
            <p className="eyebrow">The project</p>
            <h2 className="mt-4 text-3xl font-semibold tracking-[-0.04em] sm:text-5xl">
              One path from data to a playing network.
            </h2>
          </div>
          <p className="lead">
            The repository keeps the model, loader, training recipes, experiment evidence,
            export, and lc0 backend together. The site follows the same structure: start with
            the result, then open every layer of the system.
          </p>
        </div>
        <div className="mt-12 grid border-l border-t border-rule md:grid-cols-3">
          {pathways.map((pathway) => (
            <Link
              className="group border-b border-r border-rule p-6 transition-colors hover:bg-paper-deep sm:p-8"
              href={pathway.href}
              key={pathway.href}
            >
              <span className="section-number">{pathway.number}</span>
              <h3 className="mt-8 text-2xl font-semibold tracking-tight group-hover:text-cobalt">{pathway.title}</h3>
              <p className="mt-3 leading-7 text-body">{pathway.body}</p>
              <span className="mt-8 inline-block font-mono text-xs uppercase tracking-wider text-cobalt">Open →</span>
            </Link>
          ))}
        </div>
      </section>

      <section className="border-y border-rule bg-paper-deep/55 py-16 sm:py-24">
        <div className="site-shell">
          <p className="eyebrow">What the ladder says today</p>
          <div className="mt-8 grid gap-px border border-rule bg-rule md:grid-cols-3">
            <Finding value={policyElo.trend.eloPerCostDecade.toFixed(0)} unit="policy Elo / cost decade">
              A descriptive fit across the retained dense checkpoints.
            </Finding>
            <Finding value={`$${largestMeasuredCost.toFixed(2)}`} unit="largest measured run">
              The largest rated dense checkpoint, estimated from measured steady-state throughput.
            </Finding>
            <Finding value={`${(trainingDataset.samples / 1e9).toFixed(2)}B`} unit="training records">
              The current one-epoch Parquet corpus available to the final run.
            </Finding>
          </div>
        </div>
      </section>
    </main>
  );
}

function Finding({ children, unit, value }: { children: React.ReactNode; unit: string; value: string }) {
  return (
    <div className="bg-paper p-7 sm:p-9">
      <div className="text-4xl font-semibold tracking-[-0.04em] text-ink">{value}</div>
      <div className="mt-1 font-mono text-xs uppercase tracking-wider text-cobalt">{unit}</div>
      <p className="mt-6 text-sm leading-6 text-body">{children}</p>
    </div>
  );
}
