export interface PipelineStartRequest {
  models: string[]
  db_path?: string
  config_overrides?: Record<string, unknown>
}

export interface PipelineStartResponse {
  run_id: string
  sse_url: string
}

export interface DatasetInfo {
  db_path: string
  table_name: string
  row_count: number
  column_count: number
  columns: string[]
  sample_rows: Record<string, unknown>[]
}

export type AgentStatus = 'pending' | 'running' | 'completed' | 'failed'

export interface AgentState {
  name: string
  status: AgentStatus
  duration_s?: number
  metrics: Record<string, number>
}
