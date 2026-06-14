type LineChartPoint = {
  x: number;
  y: number;
};

type LineChartProps = {
  points: LineChartPoint[];
  label: string;
  xLabel: string;
  yLabel: string;
  stroke?: string;
};

export function LineChart({
  points,
  label,
  xLabel,
  yLabel,
  stroke = "#2563eb",
}: LineChartProps) {
  if (points.length === 0) {
    return null;
  }

  const width = 720;
  const height = 360;
  const padding = { top: 22, right: 28, bottom: 58, left: 76 };
  const xValues = points.map((point) => Math.log10(point.x));
  const yValues = points.map((point) => point.y);
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const minY = Math.min(...yValues);
  const maxY = Math.max(...yValues);
  const yPadding = minY === maxY ? Math.max(Math.abs(minY) * 0.05, 1) : (maxY - minY) * 0.08;
  const yDomainMin = minY >= 0 ? Math.max(0, minY - yPadding) : minY - yPadding;
  const yDomainMax = maxY + yPadding;
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;
  const xTicks = computeXTicks(points);
  const yTicks = computeYTicks(yDomainMin, yDomainMax);

  const scaleX = (value: number) =>
    padding.left + ((Math.log10(value) - minX) / Math.max(maxX - minX, 1)) * innerWidth;
  const scaleY = (value: number) =>
    height -
    padding.bottom -
    ((value - yDomainMin) / Math.max(yDomainMax - yDomainMin, 1)) * innerHeight;

  const path = points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${scaleX(point.x)} ${scaleY(point.y)}`)
    .join(" ");

  return (
    <svg
      role="img"
      aria-label={label}
      viewBox={`0 0 ${width} ${height}`}
      className="h-80 w-full overflow-visible"
    >
      {yTicks.map((tick) => (
        <g key={`y-${tick}`}>
          <line
            stroke="#e4e4e7"
            strokeWidth="1"
            x1={padding.left}
            x2={width - padding.right}
            y1={scaleY(tick)}
            y2={scaleY(tick)}
          />
          <text
            dominantBaseline="middle"
            fill="#71717a"
            fontSize="12"
            textAnchor="end"
            x={padding.left - 10}
            y={scaleY(tick)}
          >
            {formatTick(tick)}
          </text>
        </g>
      ))}
      {xTicks.map((tick) => (
        <g key={`x-${tick}`}>
          <line
            stroke="#f4f4f5"
            strokeWidth="1"
            x1={scaleX(tick)}
            x2={scaleX(tick)}
            y1={padding.top}
            y2={height - padding.bottom}
          />
          <text
            fill="#71717a"
            fontSize="12"
            textAnchor="middle"
            x={scaleX(tick)}
            y={height - padding.bottom + 24}
          >
            {formatComputeTick(tick)}
          </text>
        </g>
      ))}
      <line
        stroke="#a1a1aa"
        strokeWidth="1"
        x1={padding.left}
        x2={padding.left}
        y1={padding.top}
        y2={height - padding.bottom}
      />
      <line
        stroke="#a1a1aa"
        strokeWidth="1"
        x1={padding.left}
        x2={width - padding.right}
        y1={height - padding.bottom}
        y2={height - padding.bottom}
      />
      <text
        fill="#52525b"
        fontSize="12"
        fontWeight="600"
        textAnchor="middle"
        x={(padding.left + width - padding.right) / 2}
        y={height - 12}
      >
        {xLabel}
      </text>
      <text
        fill="#52525b"
        fontSize="12"
        fontWeight="600"
        textAnchor="middle"
        transform={`rotate(-90 ${18} ${(padding.top + height - padding.bottom) / 2})`}
        x={18}
        y={(padding.top + height - padding.bottom) / 2}
      >
        {yLabel}
      </text>
      <path d={path} fill="none" stroke={stroke} strokeLinecap="round" strokeWidth="2.5" />
      {points.map((point) => (
        <circle
          key={`${point.x}-${point.y}`}
          cx={scaleX(point.x)}
          cy={scaleY(point.y)}
          fill="white"
          r="4.5"
          stroke={stroke}
          strokeWidth="2"
        />
      ))}
    </svg>
  );
}

function computeXTicks(points: LineChartPoint[]): number[] {
  const unique = [...new Set(points.map((point) => point.x))].sort((a, b) => a - b);
  if (unique.length <= 4) {
    return unique;
  }
  return [unique[0], unique[Math.floor(unique.length / 2)], unique[unique.length - 1]];
}

function computeYTicks(min: number, max: number): number[] {
  if (min === max) {
    return [min];
  }
  const tickCount = 5;
  return Array.from({ length: tickCount }, (_, index) => min + ((max - min) * index) / 4);
}

function formatComputeTick(value: number): string {
  return `1e${Math.round(Math.log10(value))}`;
}

function formatTick(value: number): string {
  const absolute = Math.abs(value);
  if (absolute === 0) {
    return "0";
  }
  if (absolute >= 1_000_000_000) {
    return `${trim(value / 1_000_000_000)}B`;
  }
  if (absolute >= 1_000_000) {
    return `${trim(value / 1_000_000)}M`;
  }
  if (absolute >= 1_000) {
    return `${trim(value / 1_000)}K`;
  }
  if (absolute >= 10) {
    return trim(value);
  }
  return value.toFixed(2);
}

function trim(value: number): string {
  return Number(value.toFixed(2)).toString();
}
