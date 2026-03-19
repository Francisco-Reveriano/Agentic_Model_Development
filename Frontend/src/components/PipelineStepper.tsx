import type { AgentState } from '../types/pipeline'
import { CheckCircle, Circle, Loader, XCircle } from 'lucide-react'

interface Props {
  agents: AgentState[]
}

const STATUS_ICON = {
  pending: <Circle className="h-5 w-5 text-gray-600" />,
  running: <Loader className="h-5 w-5 animate-spin text-blue-400" />,
  completed: <CheckCircle className="h-5 w-5 text-green-400" />,
  failed: <XCircle className="h-5 w-5 text-red-400" />,
}

export default function PipelineStepper({ agents }: Props) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <h3 className="mb-4 text-sm font-medium text-gray-400">Pipeline Stages</h3>
      <div className="space-y-1">
        {agents.map((agent, i) => (
          <div key={agent.name} className="flex items-center gap-3 rounded p-2">
            {STATUS_ICON[agent.status]}
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate">{agent.name}</p>
              {agent.duration_s != null && (
                <p className="text-xs text-gray-500">{agent.duration_s.toFixed(1)}s</p>
              )}
            </div>
            <span className="text-xs text-gray-600">{i + 1}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
