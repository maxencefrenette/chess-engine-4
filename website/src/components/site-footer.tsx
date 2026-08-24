export function SiteFooter() {
  return (
    <footer className="mt-auto border-t border-rule py-8">
      <div className="site-shell flex flex-col gap-2 text-sm text-muted sm:flex-row sm:items-center sm:justify-between">
        <p>Chess Engine 4 · LCZero-compatible networks, trained on Modal.</p>
        <a
          className="text-link"
          href="https://github.com/maxencefrenette/chess-engine-4"
          rel="noreferrer"
          target="_blank"
        >
          Source and experiment reports ↗
        </a>
      </div>
    </footer>
  );
}
