import type { DQTest } from '../hooks/usePipelineState'

interface Props {
  dqTests: DQTest[]
}

const STATUS_STYLES: Record<string, { bg: string; text: string; border: string }> = {
  PASS: { bg: 'bg-green-500/10', text: 'text-green-400', border: 'border-green-500/20' },
  WARN: { bg: 'bg-yellow-500/10', text: 'text-yellow-400', border: 'border-yellow-500/20' },
  FAIL: { bg: 'bg-red-500/10', text: 'text-red-400', border: 'border-red-500/20' },
  PENDING: { bg: 'bg-gray-500/5', text: 'text-gray-600', border: 'border-gray-700/20' },
}

// Default DQ tests (shown as PENDING until results arrive)
const DEFAULT_TESTS = [
  { test_id: 'DQ-01', test_name: 'Completeness' },
  { test_id: 'DQ-02', test_name: 'Validity' },
  { test_id: 'DQ-03', test_name: 'Uniqueness' },
  { test_id: 'DQ-04', test_name: 'Consistency' },
  { test_id: 'DQ-05', test_name: 'Outliers' },
  { test_id: 'DQ-06', test_name: 'Distribution' },
  { test_id: 'DQ-07', test_name: 'Temporal Stability' },
  { test_id: 'DQ-08', test_name: 'Class Balance' },
  { test_id: 'DQ-09', test_name: 'Leakage Check' },
  { test_id: 'DQ-10', test_name: 'Grain Integrity' },
]

export default function DQScorecard({ dqTests }: Props) {
  const testMap = new Map(dqTests.map((t) => [t.test_id, t]))

  const passCount = dqTests.filter((t) => t.status === 'PASS').length
  const warnCount = dqTests.filter((t) => t.status === 'WARN').length
  const failCount = dqTests.filter((t) => t.status === 'FAIL').length

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-3">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xs font-medium text-gray-400 uppercase tracking-wider">DQ Scorecard</h3>
        {dqTests.length > 0 && (
          <div className="flex items-center gap-1.5 text-[10px] font-mono">
            {passCount > 0 && <span className="text-green-400">{passCount}P</span>}
            {warnCount > 0 && <span className="text-yellow-400">{warnCount}W</span>}
            {failCount > 0 && <span className="text-red-400">{failCount}F</span>}
          </div>
        )}
      </div>
      <div className="space-y-1">
        {DEFAULT_TESTS.map((dt) => {
          const result = testMap.get(dt.test_id)
          const status = result?.status || 'PENDING'
          const styles = STATUS_STYLES[status]

          return (
            <div
              key={dt.test_id}
              className={`flex items-center justify-between rounded px-2 py-1 border ${styles.bg} ${styles.border} transition-all duration-300`}
              title={result?.evidence || ''}
            >
              <div className="flex items-center gap-2 min-w-0">
                <span className="text-[9px] font-mono text-gray-500 w-8 flex-shrink-0">{dt.test_id}</span>
                <span className="text-[10px] text-gray-300 truncate">
                  {result?.test_name || dt.test_name}
                </span>
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                {result?.value && (
                  <span className="text-[9px] font-mono text-gray-500">{result.value}</span>
                )}
                <span className={`text-[9px] font-bold ${styles.text} min-w-[2rem] text-right`}>
                  {status}
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
