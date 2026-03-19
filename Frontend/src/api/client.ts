import type { DatasetInfo, PipelineStartRequest, PipelineStartResponse } from '../types/pipeline'
import type { ReportMeta } from '../types/models'

const BASE = '/api'

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, init)
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`${res.status}: ${body}`)
  }
  return res.json()
}

export const api = {
  health: () => json<{ status: string; db_connected: boolean }>('/health'),

  datasetInfo: () => json<DatasetInfo>('/dataset/info'),

  datasetPreview: (limit = 50) =>
    json<{ columns: string[]; rows: unknown[][] }>(`/dataset/preview?limit=${limit}`),

  startPipeline: (req: PipelineStartRequest) =>
    json<PipelineStartResponse>('/pipeline/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    }),

  listReports: (runId: string) => json<ReportMeta[]>(`/reports/${runId}`),

  downloadReportUrl: (runId: string, filename: string) =>
    `${BASE}/reports/${runId}/download/${filename}`,

  reportHtmlUrl: (runId: string, filename: string) =>
    `${BASE}/reports/${runId}/html/${filename}`,

  downloadReportHtmlUrl: (runId: string, filename: string) =>
    `${BASE}/reports/${runId}/download-html/${filename}`,
}
