import Link from "next/link";
import type { Metadata } from "next";
import { currentRuns, modelFamilies } from "@/data/best-runs";
import { formatDecimal } from "@/data/format";

export const metadata: Metadata = {
  title: "Experiments",
  description: "Findings, canonical scaling runs, and experiment evidence.",
};

export default function ExperimentsPage() {
  return (
    <main>
      <header className="border-b border-rule py-14 sm:py-20">
        <div className="site-shell">
          <p className="eyebrow">Experiments</p>
          <h1 className="page-title mt-5">The findings first. Every run underneath.</h1>
          <p className="lead mt-7">
            The explorer is generated from promoted experiment registries, measured throughput,
            and current recipes. Historical reports remain the evidence trail; the dashboard is
            the live view of what survived.
          </p>
        </div>
      </header>

      <section className="site-shell py-14 sm:py-20">
        <p className="eyebrow">Selected findings</p>
        <div className="mt-8 grid gap-px border border-rule bg-rule lg:grid-cols-3">
          <Insight number="01" title="Width buys predictable quality">
            The retained dense ladder traces a smooth relationship between model scale, training compute,
            validation loss, and raw policy strength.
          </Insight>
          <Insight number="02" title="Undertraining is an economic choice">
            Cheap ladder runs use fixed fractions of Chinchilla allocation so architecture and scaling choices
            can be measured without spending final-run budgets at every point.
          </Insight>
          <Insight number="03" title="Cost and FLOPs are not interchangeable">
            Hardware selection, batch size, precision, input overlap, and kernels change realized training cost
            even when model arithmetic is unchanged.
          </Insight>
        </div>
      </section>

      <section className="border-y border-rule bg-paper-deep/55 py-14 sm:py-20">
        <div className="site-shell">
          <div className="flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="eyebrow">Canonical explorer</p>
              <h2 className="mt-3 text-3xl font-semibold tracking-[-0.04em]">The measured dense family</h2>
            </div>
            <Link className="text-link font-semibold" href="/compare">Compare loss and cost →</Link>
          </div>
          <div className="mt-8 grid gap-6">
            {modelFamilies.map((family) => {
              const runs = currentRuns(family);
              const latest = runs.at(-1)!;
              return (
                <Link className="editorial-card group p-6 hover:border-cobalt sm:p-8" href={`/families/${family.id}`} key={family.id}>
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <h3 className="text-3xl font-semibold tracking-tight group-hover:text-cobalt">{family.name}</h3>
                      <p className="mt-3 max-w-lg leading-7 text-body">{family.description}</p>
                    </div>
                    <span className="font-mono text-xs text-muted">{runs.length} runs</span>
                  </div>
                  <dl className="mt-8 grid grid-cols-3 gap-4 border-t border-rule pt-5 text-sm">
                    <Term term="Largest">{latest.name}</Term>
                    <Term term="Final loss">{formatDecimal(latest.loss)}</Term>
                    <Term term="Training ratio">{family.trainingRatio}×</Term>
                  </dl>
                </Link>
              );
            })}
          </div>
        </div>
      </section>

      <section className="site-shell py-14 sm:py-20">
        <div className="grid gap-6 md:grid-cols-2">
          <Link className="editorial-card p-7 hover:border-cobalt" href="/experiments/policy-elo">
            <p className="eyebrow">Methodology</p>
            <h2 className="mt-4 text-2xl font-semibold tracking-tight">Training cost and policy Elo</h2>
            <p className="mt-3 leading-7 text-body">See how measured run costs, tournament ratings, the log-cost fit, and the BT4 target are constructed.</p>
          </Link>
          <a className="editorial-card p-7 hover:border-cobalt" href="https://github.com/maxencefrenette/chess-engine-4/tree/main/experiments" rel="noreferrer" target="_blank">
            <p className="eyebrow">Evidence archive</p>
            <h2 className="mt-4 text-2xl font-semibold tracking-tight">Experiment reports</h2>
            <p className="mt-3 leading-7 text-body">Open the dated reports, commands, raw results, plots, caveats, and promotion decisions in the repository.</p>
          </a>
        </div>
      </section>
    </main>
  );
}

function Insight({ children, number, title }: { children: React.ReactNode; number: string; title: string }) {
  return (
    <article className="bg-paper p-7 sm:p-9">
      <span className="section-number">{number}</span>
      <h2 className="mt-7 text-2xl font-semibold tracking-tight">{title}</h2>
      <p className="mt-4 text-sm leading-6 text-body">{children}</p>
    </article>
  );
}

function Term({ children, term }: { children: React.ReactNode; term: string }) {
  return (
    <div>
      <dt className="font-mono text-[10px] uppercase tracking-wider text-muted">{term}</dt>
      <dd className="mt-1 font-semibold text-ink">{children}</dd>
    </div>
  );
}
