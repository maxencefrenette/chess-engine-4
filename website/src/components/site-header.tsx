import Link from "next/link";

const navigation = [
  { href: "/", label: "Overview" },
  { href: "/how-it-works", label: "How it works" },
  { href: "/architecture", label: "Architecture" },
  { href: "/experiments", label: "Experiments" },
  { href: "/blog", label: "Blog" },
];

export function SiteHeader() {
  return (
    <header className="site-header">
      <div className="site-shell flex min-h-16 items-center justify-between gap-6 py-3">
        <Link href="/" className="flex items-center gap-3 font-semibold tracking-tight text-ink">
          <span aria-hidden="true" className="brand-mark" />
          Chess Engine 4
        </Link>
        <nav aria-label="Main navigation" className="flex items-center gap-x-5 gap-y-2 overflow-x-auto text-sm">
          {navigation.map((item) => (
            <Link className="nav-link whitespace-nowrap" href={item.href} key={item.href}>
              {item.label}
            </Link>
          ))}
          <a
            className="nav-link whitespace-nowrap"
            href="https://github.com/maxencefrenette/chess-engine-4"
            rel="noreferrer"
            target="_blank"
          >
            GitHub ↗
          </a>
        </nav>
      </div>
    </header>
  );
}
