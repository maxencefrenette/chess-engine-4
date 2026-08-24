import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "How it works",
  description: "From LCZero training records to a network running inside lc0.",
};

const stages = [
  {
    number: "01",
    title: "Read the LCZero contract",
    body: "Each record carries 112 input planes, a sparse policy target, WDL information, and moves-left supervision. The project keeps that semantic contract while changing how the records are stored and delivered.",
    detail: "Current corpus: 8,020,779,820 records across 1,203 Parquet shards.",
  },
  {
    number: "02",
    title: "Convert once, stream efficiently",
    body: "A Rust converter decodes native LCZero chunk archives into Parquet. The training loader reads complete batches with bounded prefetch and overlaps host work with the GPU at widths where it pays off.",
    detail: "The corpus is an input to training, not a service the public site depends on.",
  },
  {
    number: "03",
    title: "Derive a complete run",
    body: "A Python recipe turns model width and training ratio into architecture, batch size, learning rate, steps, precision, hardware, and input-pipeline choices. Canonical training runs execute on Modal and save periodic and final checkpoints.",
    detail: "Routine ladder runs use controlled undertraining; the planned final run uses the available one-epoch corpus deliberately.",
  },
  {
    number: "04",
    title: "Export a stable model",
    body: "A validated checkpoint is converted to a project-owned Safetensors layout. Metadata records the architecture, shape, history, activation, input normalization, heads, and source step needed by the inference backend.",
    detail: "The final release will add a published checksum and download only after end-to-end validation.",
  },
  {
    number: "05",
    title: "Run and evaluate in lc0",
    body: "The vendored lc0 fork loads the dense Safetensors model through a custom CUDA backend. Policy-only tournaments isolate the raw move distribution; searched matches and backend benchmarks answer different strength and inference questions.",
    detail: "Canonical promotion separates loss, stability, playing strength, and realized cost rather than collapsing them into one score.",
  },
];

export default function HowItWorksPage() {
  return (
    <main>
      <header className="border-b border-rule py-14 sm:py-20">
        <div className="site-shell">
          <p className="eyebrow">How it works</p>
          <h1 className="page-title mt-5">From a chess position to a network lc0 can use.</h1>
          <p className="lead mt-7">
            The system is intentionally continuous: the same data semantics and model outputs
            survive conversion, training, export, and native inference. Each boundary is tested
            because a fast network with a silent mismatch is just a fast wrong answer.
          </p>
        </div>
      </header>

      <section className="site-shell py-14 sm:py-20">
        <ol className="border-t border-rule">
          {stages.map((stage) => (
            <li className="grid gap-5 border-b border-rule py-8 md:grid-cols-[5rem_0.7fr_1.3fr] md:py-11" key={stage.number}>
              <span className="section-number">{stage.number}</span>
              <h2 className="text-2xl font-semibold tracking-tight">{stage.title}</h2>
              <div>
                <p className="leading-7 text-body">{stage.body}</p>
                <p className="mt-4 border-l-2 border-cobalt pl-4 font-mono text-xs leading-5 text-muted">{stage.detail}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <section className="border-t border-rule bg-paper-deep/55 py-14">
        <div className="site-shell grid gap-6 md:grid-cols-[1fr_auto] md:items-center">
          <div>
            <p className="eyebrow">Source of truth</p>
            <p className="mt-3 max-w-3xl leading-7 text-body">
              Public quantitative claims come from canonical experiment registries and retained reports.
              Generated JSON keeps the static site current without exposing W&amp;B or Modal credentials.
            </p>
          </div>
          <a className="text-link font-semibold" href="https://github.com/maxencefrenette/chess-engine-4" rel="noreferrer" target="_blank">
            Inspect the repository ↗
          </a>
        </div>
      </section>
    </main>
  );
}
