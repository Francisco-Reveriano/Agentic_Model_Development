import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api/client'
import type { DatasetInfo } from '../types/pipeline'
import ModelSelector from '../components/ModelSelector'

export default function HomePage() {
  const navigate = useNavigate()
  const [dataset, setDataset] = useState<DatasetInfo | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

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

  return (
    <div className="space-y-8">
      {/* Dataset Info */}
      <section className="rounded-lg border border-gray-800 bg-gray-900 p-6">
        <h2 className="mb-4 text-lg font-medium">Dataset</h2>
        {dataset ? (
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-400">Table</span>
              <p className="mt-1 font-mono">{dataset.table_name}</p>
            </div>
            <div>
              <span className="text-gray-400">Rows</span>
              <p className="mt-1 font-mono">{dataset.row_count.toLocaleString()}</p>
            </div>
            <div>
              <span className="text-gray-400">Columns</span>
              <p className="mt-1 font-mono">{dataset.column_count}</p>
            </div>
          </div>
        ) : error ? (
          <p className="text-red-400 text-sm">{error}</p>
        ) : (
          <p className="text-gray-500 text-sm">Loading dataset info...</p>
        )}
      </section>

      {/* Model Selection */}
      <ModelSelector onStart={handleStart} loading={loading} />

      {error && <p className="text-red-400 text-sm">{error}</p>}
    </div>
  )
}
