import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Architecture",
  description: "The dense LCZero-compatible Chess Engine 4 network.",
};

export default function ArchitecturePage() {
  return (
    <main>
      <header className="border-b border-rule py-14 sm:py-20">
        <div className="site-shell grid gap-8 lg:grid-cols-[0.85fr_1.15fr] lg:items-end">
          <div>
            <p className="eyebrow">Architecture</p>
            <h1 className="page-title mt-5">One chess position becomes one learned state.</h1>
          </div>
          <p className="lead">
            The model flattens LCZero&apos;s spatial input planes, projects them into a single vector,
            and refines that state through a stack of residual MLP blocks. Every parameter contributes
            to every position.
          </p>
        </div>
      </header>

      <section className="site-shell py-14 sm:py-20">
        <div className="editorial-card p-5 sm:p-8">
          <p className="eyebrow">Shared path</p>
          <div className="mt-7 grid gap-3 text-center text-sm font-semibold md:grid-cols-[1fr_auto_1fr_auto_1.4fr_auto_1fr] md:items-center">
            <ArchitectureNode label="112 LCZero planes" note="8 positions of history" />
            <Arrow />
            <ArchitectureNode label="Input projection" note="one learned state" />
            <Arrow />
            <ArchitectureNode label="8 residual MLP blocks" note="pre-norm SwiGLU" />
            <Arrow />
            <ArchitectureNode label="Three heads" note="policy · WDL · moves left" />
          </div>
        </div>

        <div className="mt-10 max-w-3xl">
          <FamilyCard label="Dense">
            <p>
              Eight pre-normalized residual SwiGLU MLP blocks. Each block expands the state to
              four times its width, then projects it back. Every weight participates in every
              network evaluation.
            </p>
            <dl className="mt-8 grid grid-cols-2 gap-5 border-t border-rule pt-6 text-sm">
              <Term term="Depth">8 blocks</Term>
              <Term term="Expansion">4×</Term>
              <Term term="N_total">All stored parameters</Term>
              <Term term="N_active">Equal to N_total</Term>
            </dl>
          </FamilyCard>
        </div>
      </section>

      <section className="border-y border-rule bg-paper-deep/55 py-14 sm:py-20">
        <div className="site-shell grid gap-8 lg:grid-cols-[0.7fr_1.3fr]">
          <div>
            <p className="eyebrow">The exact distinction</p>
            <h2 className="mt-4 text-3xl font-semibold tracking-[-0.04em]">Total is storage. Active is work.</h2>
          </div>
          <div className="space-y-5 leading-7 text-body">
            <p>
              <strong className="text-ink">N_total</strong> counts every learned parameter in the model file.
              It describes capacity and storage. <strong className="text-ink">N_active</strong> counts the parameters
              used for one position. In this dense network, every stored parameter is active for every evaluation.
            </p>
            <p>
              The two counts are therefore identical. The site keeps both labels explicit so the final model card
              remains precise and comparable with architectures whose execution may be conditional.
            </p>
          </div>
        </div>
      </section>

      <section className="site-shell py-14 sm:py-20">
        <p className="eyebrow">Final model</p>
        <div className="mt-5 max-w-3xl border-l-2 border-cobalt pl-6">
          <h2 className="text-2xl font-semibold tracking-tight">The architecture is not being declared early.</h2>
          <p className="mt-3 leading-7 text-body">
            This page will identify the final dense shape, N_total, and N_active only after selection,
            training, export, and validation are complete.
          </p>
        </div>
      </section>
    </main>
  );
}

function ArchitectureNode({ label, note }: { label: string; note: string }) {
  return (
    <div className="border border-rule bg-paper-deep/60 px-4 py-5">
      <div>{label}</div>
      <div className="mt-1 font-mono text-[10px] font-normal uppercase tracking-wider text-muted">{note}</div>
    </div>
  );
}

function Arrow() {
  return <span aria-hidden="true" className="hidden font-mono text-cobalt md:block">→</span>;
}

function FamilyCard({ children, label }: { children: React.ReactNode; label: string }) {
  return (
    <article className="editorial-card border-t-4 border-t-cobalt p-6 sm:p-8">
      <h2 className="text-3xl font-semibold tracking-[-0.04em]">{label}</h2>
      <div className="mt-5 leading-7 text-body">{children}</div>
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
