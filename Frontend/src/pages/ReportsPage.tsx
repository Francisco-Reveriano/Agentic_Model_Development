import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { api } from '../api/client'
import type { ReportMeta } from '../types/models'
import ReportCard from '../components/ReportCard'

export default function ReportsPage() {
  const { runId } = useParams<{ runId: string }>()
  const [reports, setReports] = useState<ReportMeta[]>([])
  const [error, setError] = useState('')

  useEffect(() => {
    if (!runId) return
    api.listReports(runId).then(setReports).catch((e) => setError(e.message))
  }, [runId])

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-medium">Generated Reports</h2>
      <p className="text-sm text-gray-400 font-mono">{runId}</p>

      {error && <p className="text-red-400 text-sm">{error}</p>}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        {reports.map((r) => (
          <ReportCard key={r.filename} report={r} runId={runId!} />
        ))}
      </div>

      {reports.length === 0 && !error && (
        <p className="text-gray-500 text-sm">No reports generated yet.</p>
      )}
    </div>
  )
}
