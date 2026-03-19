export interface ReportMeta {
  filename: string
  report_type: 'dq' | 'pd' | 'lgd' | 'ead' | 'el' | 'unknown'
  size_bytes: number
}

export interface ModelVersion {
  model_id: string
  model_type: string
  algorithm: string
  champion: boolean
  created_at: string
  metrics: Record<string, number>
}
