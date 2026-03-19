import { useEffect, useState } from 'react'
import { Download, FileText, Globe } from 'lucide-react'
import { api } from '../api/client'
import type { ReportMeta } from '../types/models'

const TYPE_LABELS: Record<string, string> = {
  dq: 'Data Quality Report',
  pd: 'PD Model Report',
  lgd: 'LGD Model Report',
  ead: 'EAD Model Report',
  el: 'EL Summary Report',
  unknown: 'Report',
}

const TYPE_COLORS: Record<string, string> = {
  dq: 'border-blue-500/30 bg-blue-500/5',
  pd: 'border-purple-500/30 bg-purple-500/5',
  lgd: 'border-amber-500/30 bg-amber-500/5',
  ead: 'border-orange-500/30 bg-orange-500/5',
  el: 'border-pink-500/30 bg-pink-500/5',
  unknown: 'border-gray-500/30 bg-gray-500/5',
}

const TYPE_TEXT: Record<string, string> = {
  dq: 'text-blue-400',
  pd: 'text-purple-400',
  lgd: 'text-amber-400',
  ead: 'text-orange-400',
  el: 'text-pink-400',
  unknown: 'text-gray-400',
}

interface Props {
  runId: string
}

export default function ReportViewer({ runId }: Props) {
  const [reports, setReports] = useState<ReportMeta[]>([])
  const [selectedReport, setSelectedReport] = useState<ReportMeta | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    setLoading(true)
    api
      .listReports(runId)
      .then((r) => {
        setReports(r)
        if (r.length > 0) setSelectedReport(r[0])
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [runId])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="h-8 w-8 rounded-full border-2 border-blue-500/30 border-t-blue-500 animate-spin" />
        <span className="ml-3 text-sm text-gray-400">Loading reports...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-900/50 bg-red-950/30 p-4 text-sm text-red-400">
        {error}
      </div>
    )
  }

  if (reports.length === 0) {
    return (
      <div className="rounded-lg border border-gray-800 bg-gray-900/80 p-6 text-center">
        <FileText className="mx-auto h-8 w-8 text-gray-600 mb-2" />
        <p className="text-sm text-gray-500">No reports generated yet.</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Report tabs + download buttons */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 flex-wrap">
          {reports.map((r) => (
            <button
              key={r.filename}
              onClick={() => setSelectedReport(r)}
              className={`rounded-lg border px-3 py-2 text-xs font-medium transition-all ${
                selectedReport?.filename === r.filename
                  ? `${TYPE_COLORS[r.report_type]} ${TYPE_TEXT[r.report_type]} shadow-lg`
                  : 'border-gray-700/50 text-gray-500 hover:border-gray-600 hover:text-gray-300'
              }`}
            >
              {TYPE_LABELS[r.report_type] || r.report_type}
            </button>
          ))}
        </div>

        {/* Download buttons */}
        {selectedReport && (
          <div className="flex items-center gap-2">
            <a
              href={api.downloadReportHtmlUrl(runId, selectedReport.filename)}
              className="inline-flex items-center gap-1.5 rounded-md border border-gray-700 bg-gray-800/60 px-3 py-1.5 text-xs font-medium text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
            >
              <Globe className="h-3.5 w-3.5" />
              HTML
            </a>
            <a
              href={api.downloadReportUrl(runId, selectedReport.filename)}
              className="inline-flex items-center gap-1.5 rounded-md border border-blue-500/30 bg-blue-500/10 px-3 py-1.5 text-xs font-medium text-blue-400 hover:bg-blue-500/20 hover:text-blue-300 transition-colors"
            >
              <Download className="h-3.5 w-3.5" />
              Word
            </a>
          </div>
        )}
      </div>

      {/* HTML Report iframe */}
      {selectedReport && (
        <div className="rounded-lg border border-gray-800 bg-gray-950 overflow-hidden">
          <iframe
            key={selectedReport.filename}
            src={api.reportHtmlUrl(runId, selectedReport.filename)}
            className="w-full border-0"
            style={{ height: '70vh' }}
            title={TYPE_LABELS[selectedReport.report_type] || 'Report'}
          />
        </div>
      )}
    </div>
  )
}
