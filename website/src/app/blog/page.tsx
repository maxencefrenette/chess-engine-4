import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Blog",
  description: "Technical writing from the construction of Chess Engine 4.",
};

const planned = [
  "Why a stacked MLP instead of a transformer?",
  "Building the scaling ladder",
  "A dataset format for billions of chess positions",
];

export default function BlogPage() {
  return (
    <main>
      <header className="border-b border-rule py-14 sm:py-20">
        <div className="site-shell">
          <p className="eyebrow">Blog</p>
          <h1 className="page-title mt-5">Technical notes, written one at a time.</h1>
          <p className="lead mt-7">
            The blog is where the project&apos;s judgment, dead ends, and implementation details will live.
            Each piece starts from the real decisions and evidence—not from a documentation template.
          </p>
        </div>
      </header>

      <section className="site-shell py-14 sm:py-20">
        <div className="grid gap-8 lg:grid-cols-[0.7fr_1.3fr]">
          <div>
            <p className="eyebrow">In the notebook</p>
            <h2 className="mt-4 text-3xl font-semibold tracking-[-0.04em]">The opening threads</h2>
          </div>
          <ol className="border-t border-rule">
            {planned.map((title, index) => (
              <li className="grid grid-cols-[3rem_1fr] gap-4 border-b border-rule py-6" key={title}>
                <span className="section-number">{String(index + 1).padStart(2, "0")}</span>
                <div>
                  <h3 className="text-xl font-semibold tracking-tight">{title}</h3>
                  <p className="mt-2 font-mono text-xs uppercase tracking-wider text-muted">Planned · not yet published</p>
                </div>
              </li>
            ))}
          </ol>
        </div>
      </section>
    </main>
  );
}
