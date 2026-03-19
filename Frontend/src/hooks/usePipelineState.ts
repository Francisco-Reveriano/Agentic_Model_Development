import { useMemo } from 'react'
import type { SSEEvent } from '../types/events'
import type { AgentState } from '../types/pipeline'

export function usePipelineState(events: SSEEvent[]) {
  return useMemo(() => {
    const agents: Record<string, AgentState> = {}
    const logs: Array<{ agent: string; content: string; type: string; ts: number }> = []
    const metrics: Record<string, Record<string, number>> = {}
    let pipelineStatus: 'idle' | 'running' | 'completed' | 'failed' = 'idle'
    let reports: string[] = []

    for (const event of events) {
      const d = event.data as Record<string, unknown>
      const agent = (d.agent as string) || ''

      switch (event.type) {
        case 'pipeline_start':
          pipelineStatus = 'running'
          for (const a of (d.agents as string[]) || []) {
            agents[a] = { name: a, status: 'pending', metrics: {} }
          }
          break

        case 'agent_start':
          if (agent) {
            agents[agent] = { ...agents[agent], name: agent, status: 'running', metrics: agents[agent]?.metrics || {} }
          }
          break

        case 'agent_log':
          logs.push({
            agent,
            content: d.content as string,
            type: d.type as string,
            ts: (d.timestamp as number) || Date.now() / 1000,
          })
          break

        case 'agent_metric':
          if (!metrics[agent]) metrics[agent] = {}
          metrics[agent][d.metric as string] = d.value as number
          if (agents[agent]) {
            agents[agent].metrics = { ...agents[agent].metrics, [d.metric as string]: d.value as number }
          }
          break

        case 'agent_complete':
          if (agents[agent]) {
            agents[agent] = {
              ...agents[agent],
              status: d.status === 'success' ? 'completed' : 'failed',
              duration_s: d.duration_s as number,
            }
          }
          break

        case 'agent_error':
          if (agents[agent]) {
            agents[agent] = { ...agents[agent], status: 'failed' }
          }
          break

        case 'pipeline_complete':
          pipelineStatus = (d.status as 'completed' | 'failed') || 'completed'
          reports = (d.reports as string[]) || []
          break

        case 'pipeline_error':
          pipelineStatus = 'failed'
          break
      }
    }

    return {
      agents: Object.values(agents),
      logs,
      metrics,
      pipelineStatus,
      reports,
    }
  }, [events])
}
