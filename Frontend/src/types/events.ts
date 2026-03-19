export type SSEEventType =
  | 'agent_start'
  | 'agent_log'
  | 'agent_metric'
  | 'agent_complete'
  | 'agent_error'
  | 'agent_substep'
  | 'agent_table'
  | 'agent_chart_data'
  | 'dq_scorecard_update'
  | 'pipeline_start'
  | 'pipeline_complete'
  | 'pipeline_error'
  | 'tournament_start'
  | 'model_trained'
  | 'phase_complete'
  | 'feature_consensus'
  | 'iteration_update'
  | 'model_pruned'
  | 'champion_declared'
  | 'heartbeat'
  | 'stream_end'

export interface SSEEvent {
  type: SSEEventType
  data: Record<string, unknown>
  timestamp?: number
}

export interface AgentStartData {
  agent: string
  stage: number
  total_stages: number
}

export interface AgentLogData {
  agent: string
  content: string
  type: 'text' | 'tool_call' | 'tool_result'
  tool_name?: string
}

export interface AgentMetricData {
  agent: string
  metric: string
  value: number
}

export interface AgentCompleteData {
  agent: string
  status: 'success' | 'failed'
  duration_s: number
}

export interface AgentSubstepData {
  agent: string
  substep: string
  status: 'running' | 'completed'
}

export interface AgentTableData {
  agent: string
  table_name: string
  columns: string[]
  rows: unknown[][]
}

export interface DQTestData {
  agent: string
  test_id: string
  test_name: string
  status: 'PASS' | 'WARN' | 'FAIL'
  value: string
  threshold: string
  evidence: string
}

export interface ChartDataEvent {
  agent: string
  chart_name: string
  data: Record<string, unknown>[]
}

export interface PipelineCompleteData {
  run_id: string
  status: 'completed' | 'failed'
  completed_agents: string[]
  total_duration_s: number
  reports?: string[]
  error?: string
}

export interface ModelTrainedData {
  model: string
  rank: number
  primary_metric: number
  time_s: number
}

export interface ChampionDeclaredData {
  champion: string
  score: number
  runner_up: string
}
