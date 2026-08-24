type MetricCardProps = {
  label: string;
  value: string;
  detail?: string;
};

export function MetricCard({ label, value, detail }: MetricCardProps) {
  return (
    <div className="editorial-card p-4">
      <div className="text-sm font-medium text-muted">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-ink">{value}</div>
      {detail ? <div className="mt-2 text-sm text-muted">{detail}</div> : null}
    </div>
  );
}
