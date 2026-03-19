import { useParams } from 'react-router-dom'
import ReportViewer from '../components/ReportViewer'

export default function ReportsPage() {
  const { runId } = useParams<{ runId: string }>()

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-medium">Generated Reports</h2>
        <p className="text-sm text-gray-400 font-mono">{runId}</p>
      </div>
      {runId && <ReportViewer runId={runId} />}
    </div>
  )
}
