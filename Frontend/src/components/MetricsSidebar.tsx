interface Props {
  metrics: Record<string, Record<string, number>>
}

const METRIC_LABELS: Record<string, string> = {
  auc: 'AUC-ROC',
  gini: 'Gini',
  ks: 'KS Stat',
  brier: 'Brier',
  psi: 'PSI',
  rmse: 'RMSE',
  mae: 'MAE',
  r2: 'R²',
  row_count: 'Rows',
  default_rate: 'Default Rate',
}

function fmt(v: number): string {
  if (v > 1000) return v.toLocaleString()
  return v.toFixed(4)
}

export default function MetricsSidebar({ metrics }: Props) {
  const agents = Object.keys(metrics)

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <h3 className="mb-4 text-sm font-medium text-gray-400">Metrics</h3>
      {agents.length === 0 && (
        <p className="text-xs text-gray-600">No metrics yet</p>
      )}
      {agents.map((agent) => (
        <div key={agent} className="mb-4">
          <p className="mb-2 text-xs font-medium text-gray-500">{agent}</p>
          <div className="space-y-1">
            {Object.entries(metrics[agent]).map(([k, v]) => (
              <div key={k} className="flex justify-between text-xs">
                <span className="text-gray-400">{METRIC_LABELS[k] || k}</span>
                <span className="font-mono">{fmt(v)}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}
