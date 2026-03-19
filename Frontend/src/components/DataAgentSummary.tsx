interface Props {
  metrics: Record<string, number>
  chartData: Record<string, unknown[]>
}

interface CardDef {
  key: string
  label: string
  format: (v: number) => string
  color: string
}

const CARDS: CardDef[] = [
  { key: 'row_count', label: 'Total Rows', format: (v) => v >= 1e6 ? `${(v / 1e6).toFixed(2)}M` : v.toLocaleString(), color: 'text-blue-400' },
  { key: 'resolved_rows', label: 'Resolved', format: (v) => v >= 1e6 ? `${(v / 1e6).toFixed(2)}M` : v.toLocaleString(), color: 'text-emerald-400' },
  { key: 'default_rate', label: 'Default Rate', format: (v) => `${(v * 100).toFixed(1)}%`, color: 'text-amber-400' },
  { key: 'feature_count', label: 'Features', format: (v) => v.toString(), color: 'text-purple-400' },
  { key: 'default_count', label: 'Defaults', format: (v) => v.toLocaleString(), color: 'text-red-400' },
  { key: 'fully_paid_count', label: 'Fully Paid', format: (v) => v.toLocaleString(), color: 'text-green-400' },
]

function Sparkline({ data }: { data: { default_rate: number }[] }) {
  if (!data || data.length < 2) return null
  const rates = data.map((d) => d.default_rate).filter((r) => r != null && r > 0)
  if (rates.length < 2) return null

  const width = 140
  const height = 36
  const padding = 2
  const min = Math.min(...rates)
  const max = Math.max(...rates)
  const range = max - min || 0.01

  const points = rates
    .map((r, i) => {
      const x = padding + (i / (rates.length - 1)) * (width - 2 * padding)
      const y = height - padding - ((r - min) / range) * (height - 2 * padding)
      return `${x},${y}`
    })
    .join(' ')

  return (
    <div className="mt-2">
      <p className="text-[9px] text-gray-500 mb-1">Default Rate by Vintage</p>
      <svg width={width} height={height} className="overflow-visible">
        <defs>
          <linearGradient id="sparkGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgb(251, 191, 36)" stopOpacity="0.3" />
            <stop offset="100%" stopColor="rgb(251, 191, 36)" stopOpacity="0" />
          </linearGradient>
        </defs>
        <polygon
          points={`${padding},${height - padding} ${points} ${width - padding},${height - padding}`}
          fill="url(#sparkGrad)"
        />
        <polyline
          points={points}
          fill="none"
          stroke="rgb(251, 191, 36)"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
      <div className="flex justify-between text-[8px] text-gray-600 font-mono">
        <span>{(min * 100).toFixed(1)}%</span>
        <span>{(max * 100).toFixed(1)}%</span>
      </div>
    </div>
  )
}

export default function DataAgentSummary({ metrics, chartData }: Props) {
  const hasMetrics = Object.keys(metrics).length > 0
  const vintageData = (chartData['vintage_default_rates'] || []) as { default_rate: number }[]

  if (!hasMetrics && vintageData.length === 0) return null

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-3">
      <h3 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">Data Summary</h3>
      <div className="grid grid-cols-2 gap-1.5">
        {CARDS.map((card) => {
          const val = metrics[card.key]
          if (val == null) return null
          return (
            <div key={card.key} className="rounded-md bg-gray-800/50 px-2 py-1.5 text-center">
              <p className="text-[9px] text-gray-500">{card.label}</p>
              <p className={`text-sm font-mono font-bold ${card.color}`}>
                {card.format(val)}
              </p>
            </div>
          )
        })}
      </div>
      <Sparkline data={vintageData} />
    </div>
  )
}
