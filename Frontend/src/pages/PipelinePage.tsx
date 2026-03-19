import { useMemo } from 'react'
import { useParams } from 'react-router-dom'
import { useSSE } from '../hooks/useSSE'
import { usePipelineState } from '../hooks/usePipelineState'
import PipelineStepper from '../components/PipelineStepper'
import StreamingLog from '../components/StreamingLog'
import ToolTracker from '../components/ToolTracker'
import DQScorecard from '../components/DQScorecard'
import DataAgentSummary from '../components/DataAgentSummary'
import ReportViewer from '../components/ReportViewer'

export default function PipelinePage() {
  const { runId } = useParams<{ runId: string }>()
  const { events, status } = useSSE(runId ? `/api/pipeline/stream/${runId}` : null)
  const state = usePipelineState(events)

  // Detect if Data_Agent is currently running or just completed
  const dataAgentActive = useMemo(() => {
    const da = state.agents.find((a) => a.name === 'Data_Agent')
    return da?.status === 'running' || da?.status === 'completed'
  }, [state.agents])

  // Collect Data_Agent metrics from all sources
  const dataAgentMetrics = useMemo(() => {
    return state.metrics['Data_Agent'] || {}
  }, [state.metrics])

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-medium">Pipeline Execution</h2>
          <p className="text-sm text-gray-400 font-mono">{runId}</p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-500">SSE: {status}</span>
          {state.pipelineStatus === 'completed' && (
            <span className="rounded-full bg-green-500/10 border border-green-500/20 px-3 py-1 text-xs font-medium text-green-400">
              Pipeline Complete
            </span>
          )}
          {state.pipelineStatus === 'failed' && (
            <span className="rounded-full bg-red-500/10 border border-red-500/20 px-3 py-1 text-xs font-medium text-red-400">
              Pipeline Failed
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-12 gap-4" style={{ height: 'calc(100vh - 220px)' }}>
        {/* Left: Pipeline Stepper + DQ Scorecard */}
        <div className="col-span-2 flex flex-col gap-3 overflow-auto min-h-0">
          <PipelineStepper agents={state.agents} substeps={state.substeps} />
          {dataAgentActive && state.dqTests.length > 0 && (
            <DQScorecard dqTests={state.dqTests} />
          )}
        </div>

        {/* Center: Streaming Log */}
        <div className="col-span-7 min-h-0">
          <StreamingLog logs={state.logs} />
        </div>

        {/* Right: Data Summary + Tool Tracker + Metrics */}
        <div className="col-span-3 min-h-0 overflow-auto flex flex-col gap-3">
          {dataAgentActive && (
            <DataAgentSummary metrics={dataAgentMetrics} chartData={state.chartData} />
          )}
          <ToolTracker
            toolCalls={state.toolCalls}
            totalToolCalls={state.totalToolCalls}
            metrics={state.metrics}
          />
        </div>
      </div>

      {/* Report viewer — shown when pipeline completes */}
      {state.pipelineStatus === 'completed' && runId && (
        <div className="pt-2">
          <h2 className="text-lg font-medium mb-4">Generated Reports</h2>
          <ReportViewer runId={runId} />
        </div>
      )}
    </div>
  )
}
