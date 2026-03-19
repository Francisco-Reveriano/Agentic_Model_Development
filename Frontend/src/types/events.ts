export type SSEEventType =
  | 'agent_start'
  | 'agent_log'
  | 'agent_metric'
  | 'agent_complete'
  | 'agent_error'
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
