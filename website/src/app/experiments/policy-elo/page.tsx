import type { Metadata } from "next";
import { PolicyEloChart } from "@/components/policy-elo-chart";
import { policyElo } from "@/data/best-runs";

export const metadata: Metadata = {
  title: "Training cost and policy Elo",
  description: "Methodology for the Chess Engine 4 cost-strength chart.",
};

export default function PolicyEloMethodPage() {
  const targetCost = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(policyElo.trend.estimatedTargetCost);

  return (
    <main>
      <header className="border-b border-rule py-14 sm:py-20">
        <div className="site-shell">
          <p className="eyebrow">Experiment methodology</p>
          <h1 className="page-title mt-5">Training cost and policy Elo</h1>
          <p className="lead mt-7">
            This chart asks a narrow question: as supervised training cost grows, how much stronger
            is the network&apos;s raw move policy? It does not measure searched engine strength.
          </p>
        </div>
      </header>

      <section className="site-shell py-10 sm:py-16">
        <div className="editorial-card p-4 sm:p-8">
          <PolicyEloChart data={policyElo} />
        </div>
        <p className="mt-5 max-w-4xl text-sm leading-6 text-muted">
          Observed CE4 points use canonical checkpoints and measured steady-state throughput. The dashed
          line is one linear fit against log10 cost across all {policyElo.points.length} dense observations. Its intersection with BT4
          is about {targetCost}; extending that fit several cost decades is deliberately shown as an
          extrapolation, not a budget forecast.
        </p>
      </section>

      <section className="border-y border-rule bg-paper-deep/55 py-14 sm:py-20">
        <div className="site-shell grid gap-8 md:grid-cols-2">
          <Method title="Training cost">
            Runtime is derived from each canonical sample count, batch size, and measured wall time per step.
            Dollars use the selected Modal GPU plus eight CPU cores. Startup, compilation, storage, and artifact
            transfer are excluded, so these are comparable steady-state estimates rather than invoices.
          </Method>
          <Method title="Policy Elo">
            Four adaptive waves use mirrored two-move openings and an effective policy batch of 256. The engine
            plays the highest-policy legal move without MCTS. The common 14-network field is fitted jointly, then
            the displayed ratings are shifted so BT4 is exactly zero.
          </Method>
          <Method title="BT4 target">
            BT4 comes from the same retained tournament, so the horizontal line shares the protocol and rating
            origin. The chart makes no claim about what BT4 cost to train and does not assign it an x-coordinate.
          </Method>
        </div>
      </section>

      <section className="site-shell py-14 sm:py-20">
        <div className="grid gap-8 lg:grid-cols-[0.7fr_1.3fr]">
          <div>
            <p className="eyebrow">Read this carefully</p>
            <h2 className="mt-4 text-3xl font-semibold tracking-[-0.04em]">What the line does not know</h2>
          </div>
          <ul className="space-y-4 leading-7 text-body">
            <li>Future architectures may move the frontier instead of following it.</li>
            <li>Policy Elo is sensitive to the fixed batching and opening protocol and is not transferable as an absolute rating.</li>
            <li>Marginal Elo intervals are material; the homepage suppresses them for legibility, while the retained report records them.</li>
            <li>Dollar estimates depend on dated hardware prices and measured throughput and should be regenerated when either changes.</li>
          </ul>
        </div>
        <div className="mt-10 flex flex-wrap gap-5 text-sm font-semibold">
          <a className="text-link" href="https://github.com/maxencefrenette/chess-engine-4/tree/main/experiments/2026-08-07.01-dense-moe-policy-elo" rel="noreferrer" target="_blank">Tournament report and raw JSON ↗</a>
          <a className="text-link" href="https://github.com/maxencefrenette/chess-engine-4/tree/main/experiments/2026-08-08.01-paired-elo-confidence" rel="noreferrer" target="_blank">Confidence-interval methodology ↗</a>
        </div>
      </section>
    </main>
  );
}

function Method({ children, title }: { children: React.ReactNode; title: string }) {
  return (
    <article>
      <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
      <p className="mt-4 text-sm leading-6 text-body">{children}</p>
    </article>
  );
}
