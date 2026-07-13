type SparklineProps = {
  points: { x: number; y: number }[];
  stalePoints: { x: number; y: number }[];
  label: string;
  stroke?: string;
};

export function Sparkline({
  points,
  stalePoints,
  label,
  stroke = "#2563eb",
}: SparklineProps) {
  if (points.length === 0) {
    return null;
  }

  const width = 220;
  const height = 72;
  const padding = 6;
  const allPoints = [...points, ...stalePoints];
  const xValues = allPoints.map((point) => Math.log10(point.x));
  const yValues = allPoints.map((point) => point.y);
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);

  const scaleX = (value: number) =>
    padding + ((Math.log10(value) - minX) / Math.max(maxX - minX, 1)) * (width - padding * 2);
  const scaleY = (value: number) =>
    height -
    padding -
    ((value - minY) / Math.max(maxY - minY, 1)) * (height - padding * 2);

  const path = points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${scaleX(point.x)} ${scaleY(point.y)}`)
    .join(" ");

  return (
    <svg
      role="img"
      aria-label={label}
      viewBox={`0 0 ${width} ${height}`}
      className="h-20 w-full overflow-visible"
    >
      <path d={path} fill="none" stroke={stroke} strokeLinecap="round" strokeWidth="2" />
      {points.map((point) => (
        <circle
          key={`${point.x}-${point.y}`}
          cx={scaleX(point.x)}
          cy={scaleY(point.y)}
          fill="white"
          r="3.5"
          stroke={stroke}
          strokeWidth="1.5"
        />
      ))}
      {stalePoints.map((point) => (
        <circle
          key={`stale-${point.x}-${point.y}`}
          cx={scaleX(point.x)}
          cy={scaleY(point.y)}
          fill="#d4d4d8"
          opacity="0.8"
          r="3.5"
          stroke="#a1a1aa"
          strokeWidth="1.5"
        />
      ))}
    </svg>
  );
}
