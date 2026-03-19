import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { DatasetInfo } from '../types/pipeline'
import ModelSelector from '../components/ModelSelector'

export default function HomePage() {
  const navigate = useNavigate()
  const [dataset, setDataset] = useState<DatasetInfo | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [columnFilter, setColumnFilter] = useState('')

  useEffect(() => {
    api.datasetInfo().then(setDataset).catch((e) => setError(e.message))
  }, [])

  const handleStart = async (models: string[]) => {
    setLoading(true)
    setError('')
    try {
      const res = await api.startPipeline({ models })
      navigate(`/pipeline/${res.run_id}`)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to start pipeline')
      setLoading(false)
    }
  }

  const filteredColumns = useMemo(() => {
    if (!dataset) return []
    if (!columnFilter.trim()) return dataset.columns
    const q = columnFilter.toLowerCase()
    return dataset.columns.filter((c) => c.toLowerCase().includes(q))
  }, [dataset, columnFilter])

  const formatCell = (value: unknown): string => {
    if (value === null || value === undefined) return ''
    if (typeof value === 'number') {
      if (Number.isInteger(value)) return value.toLocaleString()
      return value.toFixed(2)
    }
    return String(value)
  }

  return (
    <div className="grid grid-cols-12 gap-5 h-[calc(100vh-80px)]">
      {/* Left Panel — 1/3 */}
      <div className="col-span-4 flex flex-col gap-5 overflow-auto pr-1">
        <ModelSelector onStart={handleStart} loading={loading} />

        {/* Dataset Stats */}
        {dataset && (
          <section className="rounded-lg border border-gray-800 bg-gray-900/80 backdrop-blur p-5">
            <h2 className="mb-4 text-sm font-medium text-gray-400 uppercase tracking-wider">Dataset</h2>
            <div className="grid grid-cols-3 gap-3">
              <div className="rounded-md bg-gray-800/60 p-3 text-center">
                <p className="text-xs text-gray-500 mb-1">Table</p>
                <p className="font-mono text-sm text-gray-200">{dataset.table_name}</p>
              </div>
              <div className="rounded-md bg-gray-800/60 p-3 text-center">
                <p className="text-xs text-gray-500 mb-1">Rows</p>
                <p className="font-mono text-sm text-blue-400">{dataset.row_count.toLocaleString()}</p>
              </div>
              <div className="rounded-md bg-gray-800/60 p-3 text-center">
                <p className="text-xs text-gray-500 mb-1">Columns</p>
                <p className="font-mono text-sm text-emerald-400">{dataset.column_count}</p>
              </div>
            </div>

            {/* Column list */}
            <div className="mt-4">
              <p className="text-xs text-gray-500 mb-2">Column Types</p>
              <div className="flex flex-wrap gap-1">
                {['id', 'loan_amnt', 'term', 'int_rate', 'grade', 'loan_status', 'annual_inc', 'dti'].map((col) => (
                  <span key={col} className="inline-block rounded bg-gray-800 px-2 py-0.5 text-[10px] font-mono text-gray-400">
                    {col}
                  </span>
                ))}
                {dataset.column_count > 8 && (
                  <span className="inline-block rounded bg-gray-800/50 px-2 py-0.5 text-[10px] text-gray-500">
                    +{dataset.column_count - 8} more
                  </span>
                )}
              </div>
            </div>
          </section>
        )}

        {error && (
          <div className="rounded-lg border border-red-900/50 bg-red-950/30 p-3 text-sm text-red-400">
            {error}
          </div>
        )}
      </div>

      {/* Right Panel — 2/3 */}
      <div className="col-span-8 flex flex-col min-h-0">
        <section className="rounded-lg border border-gray-800 bg-gray-900/80 backdrop-blur flex flex-col flex-1 min-h-0">
          {/* Header bar */}
          <div className="flex items-center justify-between px-5 py-3 border-b border-gray-800">
            <div className="flex items-center gap-3">
              <h2 className="text-sm font-medium text-gray-300">Data Preview</h2>
              {dataset && (
                <div className="flex items-center gap-2">
                  <span className="rounded-full bg-blue-500/10 border border-blue-500/20 px-2.5 py-0.5 text-[10px] font-mono text-blue-400">
                    {dataset.sample_rows.length} rows
                  </span>
                  <span className="rounded-full bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-0.5 text-[10px] font-mono text-emerald-400">
                    {filteredColumns.length}/{dataset.column_count} cols
                  </span>
                </div>
              )}
            </div>
            {/* Column filter */}
            <div className="relative">
              <svg className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                type="text"
                placeholder="Filter columns..."
                value={columnFilter}
                onChange={(e) => setColumnFilter(e.target.value)}
                className="rounded-md border border-gray-700 bg-gray-800/60 py-1.5 pl-8 pr-3 text-xs text-gray-300 placeholder-gray-600 outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/20 w-48 transition-all"
              />
            </div>
          </div>

          {/* Table */}
          {dataset ? (
            <div className="flex-1 overflow-auto min-h-0">
              <table className="w-full text-xs font-mono border-collapse">
                <thead className="sticky top-0 z-10">
                  <tr className="bg-gradient-to-r from-gray-800 to-gray-800/90">
                    <th className="px-3 py-2.5 text-left text-[10px] font-semibold text-gray-500 uppercase tracking-wider border-b border-gray-700 w-10">#</th>
                    {filteredColumns.map((col) => (
                      <th
                        key={col}
                        className="px-3 py-2.5 text-left text-[10px] font-semibold text-gray-400 uppercase tracking-wider border-b border-gray-700 whitespace-nowrap"
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dataset.sample_rows.map((row, i) => (
                    <tr
                      key={i}
                      className="border-b border-gray-800/50 transition-colors hover:bg-blue-500/[0.03]"
                    >
                      <td className="px-3 py-1.5 text-gray-600 tabular-nums">{i + 1}</td>
                      {filteredColumns.map((col) => {
                        const val = row[col]
                        const isNull = val === null || val === undefined || val === ''
                        return (
                          <td
                            key={col}
                            className={`px-3 py-1.5 whitespace-nowrap max-w-[180px] truncate tabular-nums ${
                              isNull ? 'text-gray-700 italic' : 'text-gray-300'
                            }`}
                            title={isNull ? 'null' : String(val)}
                          >
                            {isNull ? 'null' : formatCell(val)}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <div className="flex flex-col items-center gap-3">
                <div className="h-8 w-8 rounded-full border-2 border-blue-500/30 border-t-blue-500 animate-spin" />
                <p className="text-sm text-gray-500">Loading dataset...</p>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
