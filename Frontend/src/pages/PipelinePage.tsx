import { useParams, useNavigate } from 'react-router-dom'
import { useSSE } from '../hooks/useSSE'
import { usePipelineState } from '../hooks/usePipelineState'
import PipelineStepper from '../components/PipelineStepper'
import StreamingLog from '../components/StreamingLog'
import MetricsSidebar from '../components/MetricsSidebar'

export default function PipelinePage() {
  const { runId } = useParams<{ runId: string }>()
  const navigate = useNavigate()
  const { events, status } = useSSE(runId ? `/api/pipeline/stream/${runId}` : null)
  const state = usePipelineState(events)

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-medium">Pipeline Execution</h2>
          <p className="text-sm text-gray-400 font-mono">{runId}</p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-500">
            SSE: {status}
          </span>
          {state.pipelineStatus === 'completed' && (
            <button
              onClick={() => navigate(`/reports/${runId}`)}
              className="rounded bg-green-700 px-4 py-2 text-sm font-medium hover:bg-green-600"
            >
              View Reports
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-12 gap-6">
        {/* Stepper */}
        <div className="col-span-3">
          <PipelineStepper agents={state.agents} />
        </div>

        {/* Streaming Log */}
        <div className="col-span-6">
          <StreamingLog logs={state.logs} />
        </div>

        {/* Metrics Sidebar */}
        <div className="col-span-3">
          <MetricsSidebar metrics={state.metrics} />
        </div>
      </div>
    </div>
  )
}
