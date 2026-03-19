import { Download, FileText } from 'lucide-react'
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

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

interface Props {
  report: ReportMeta
  runId: string
}

export default function ReportCard({ report, runId }: Props) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-5">
      <div className="mb-3 flex items-center gap-2">
        <FileText className="h-5 w-5 text-blue-400" />
        <h3 className="font-medium">{TYPE_LABELS[report.report_type] || report.report_type}</h3>
      </div>
      <p className="mb-1 text-sm text-gray-400 font-mono truncate">{report.filename}</p>
      <p className="mb-4 text-xs text-gray-500">{formatBytes(report.size_bytes)}</p>
      <a
        href={api.downloadReportUrl(runId, report.filename)}
        className="inline-flex items-center gap-2 rounded bg-blue-600 px-4 py-2 text-sm font-medium hover:bg-blue-500"
      >
        <Download className="h-4 w-4" />
        Download
      </a>
    </div>
  )
}
