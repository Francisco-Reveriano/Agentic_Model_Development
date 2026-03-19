import type { ToolCall } from '../hooks/usePipelineState'

interface Props {
  toolCalls: ToolCall[]
  totalToolCalls: number
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
  r2: 'R\u00B2',
  row_count: 'Rows',
  default_rate: 'Default Rate',
}

function fmt(v: number): string {
  if (v > 1000) return v.toLocaleString()
  return v.toFixed(4)
}

const AGENT_COLORS: Record<string, string> = {
  Data_Agent: 'text-blue-400',
  Feature_Agent: 'text-emerald-400',
  PD_Agent: 'text-purple-400',
  LGD_Agent: 'text-amber-400',
  EAD_Agent: 'text-orange-400',
  EL_Agent: 'text-pink-400',
  Report_Agent: 'text-cyan-400',
}

export default function ToolTracker({ toolCalls, totalToolCalls, metrics }: Props) {
  const metricAgents = Object.keys(metrics)

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Tool Calls Panel */}
      <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 flex-1 min-h-0 flex flex-col">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-gray-400">Tool Calls</h3>
          <div className="flex items-center gap-1.5">
            <div className="h-2 w-2 rounded-full bg-yellow-500 animate-pulse" />
            <span className="text-sm font-mono font-bold text-yellow-400">{totalToolCalls}</span>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto space-y-1 min-h-0">
          {toolCalls.length === 0 && (
            <p className="text-xs text-gray-600 italic">No tools called yet</p>
          )}
          {toolCalls.map((tc) => (
            <div
              key={`${tc.agent}::${tc.name}`}
              className="flex items-center justify-between rounded-md bg-gray-800/50 px-2.5 py-1.5 group hover:bg-gray-800 transition-colors"
            >
              <div className="flex items-center gap-2 min-w-0">
                <svg className="h-3 w-3 text-yellow-500/70 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                <span className="text-xs font-mono text-gray-300 truncate">{tc.name}</span>
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                <span className={`text-[9px] ${AGENT_COLORS[tc.agent] || 'text-gray-500'}`}>
                  {tc.agent.replace('_Agent', '')}
                </span>
                <span className="inline-flex items-center justify-center min-w-[1.25rem] h-5 rounded-full bg-yellow-500/15 border border-yellow-500/20 px-1.5 text-[10px] font-bold font-mono text-yellow-400">
                  {tc.count}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Metrics Panel */}
      <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
        <h3 className="mb-3 text-sm font-medium text-gray-400">Metrics</h3>
        {metricAgents.length === 0 && (
          <p className="text-xs text-gray-600 italic">No metrics yet</p>
        )}
        {metricAgents.map((agent) => (
          <div key={agent} className="mb-3 last:mb-0">
            <p className={`mb-1.5 text-xs font-medium ${AGENT_COLORS[agent] || 'text-gray-500'}`}>{agent}</p>
            <div className="space-y-1">
              {Object.entries(metrics[agent]).map(([k, v]) => (
                <div key={k} className="flex justify-between text-xs">
                  <span className="text-gray-400">{METRIC_LABELS[k] || k}</span>
                  <span className="font-mono text-gray-200">{fmt(v)}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
