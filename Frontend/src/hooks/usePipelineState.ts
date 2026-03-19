import { useMemo } from 'react'
import type { SSEEvent } from '../types/events'
import type { AgentState } from '../types/pipeline'

export interface ToolCall {
  name: string
  count: number
  agent: string
  lastTs: number
}

export interface DQTest {
  test_id: string
  test_name: string
  status: 'PASS' | 'WARN' | 'FAIL' | 'PENDING'
  value: string
  threshold: string
  evidence: string
}

export interface Substep {
  name: string
  status: 'running' | 'completed'
}

export interface TableEntry {
  agent: string
  table_name: string
  columns: string[]
  rows: unknown[][]
}

export function usePipelineState(events: SSEEvent[]) {
  return useMemo(() => {
    const agents: Record<string, AgentState> = {}
    const logs: Array<{ agent: string; content: string; type: string; ts: number; tableData?: TableEntry }> = []
    const metrics: Record<string, Record<string, number>> = {}
    const toolCalls: Record<string, ToolCall> = {}
    let totalToolCalls = 0
    let pipelineStatus: 'idle' | 'running' | 'completed' | 'failed' = 'idle'
    let reports: string[] = []
    const dqTests: Record<string, DQTest> = {}
    const substeps: Record<string, Substep[]> = {}
    const chartData: Record<string, unknown[]> = {}

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

        case 'agent_log': {
          const logType = d.type as string
          if (logType === 'tool_call') {
            const toolName = (d.tool_name as string) || (d.content as string).replace('Calling tool: ', '')
            const key = `${agent}::${toolName}`
            if (toolCalls[key]) {
              toolCalls[key].count += 1
              toolCalls[key].lastTs = (d.timestamp as number) || Date.now() / 1000
            } else {
              toolCalls[key] = { name: toolName, count: 1, agent, lastTs: (d.timestamp as number) || Date.now() / 1000 }
            }
            totalToolCalls += 1
          } else if (logType === 'tool_result') {
            // Skip
          } else {
            logs.push({
              agent,
              content: d.content as string,
              type: logType,
              ts: (d.timestamp as number) || Date.now() / 1000,
            })
          }
          break
        }

        case 'agent_table': {
          const tableEntry: TableEntry = {
            agent,
            table_name: d.table_name as string,
            columns: d.columns as string[],
            rows: d.rows as unknown[][],
          }
          logs.push({
            agent,
            content: d.table_name as string,
            type: 'table',
            ts: (d.timestamp as number) || Date.now() / 1000,
            tableData: tableEntry,
          })
          break
        }

        case 'dq_scorecard_update':
          dqTests[d.test_id as string] = {
            test_id: d.test_id as string,
            test_name: d.test_name as string,
            status: d.status as 'PASS' | 'WARN' | 'FAIL',
            value: d.value as string,
            threshold: d.threshold as string,
            evidence: (d.evidence as string) || '',
          }
          break

        case 'agent_substep': {
          if (!substeps[agent]) substeps[agent] = []
          const existing = substeps[agent].find((s) => s.name === (d.substep as string))
          if (existing) {
            existing.status = d.status as 'running' | 'completed'
          } else {
            substeps[agent].push({ name: d.substep as string, status: d.status as 'running' | 'completed' })
          }
          break
        }

        case 'agent_chart_data':
          chartData[d.chart_name as string] = d.data as unknown[]
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
      toolCalls: Object.values(toolCalls).sort((a, b) => b.lastTs - a.lastTs),
      totalToolCalls,
      pipelineStatus,
      reports,
      dqTests: Object.values(dqTests),
      substeps,
      chartData,
    }
  }, [events])
}
