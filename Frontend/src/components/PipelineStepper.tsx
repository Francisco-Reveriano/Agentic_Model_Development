import type { AgentState } from '../types/pipeline'
import type { Substep } from '../hooks/usePipelineState'
import { CheckCircle, Circle, Loader, XCircle } from 'lucide-react'

interface Props {
  agents: AgentState[]
  substeps?: Record<string, Substep[]>
}

const STATUS_ICON = {
  pending: <Circle className="h-4 w-4 text-gray-600" />,
  running: <Loader className="h-4 w-4 animate-spin text-blue-400" />,
  completed: <CheckCircle className="h-4 w-4 text-green-400" />,
  failed: <XCircle className="h-4 w-4 text-red-400" />,
}

const SUBSTEP_ICON = {
  running: <Loader className="h-2.5 w-2.5 animate-spin text-blue-400" />,
  completed: <CheckCircle className="h-2.5 w-2.5 text-green-500" />,
}

export default function PipelineStepper({ agents, substeps = {} }: Props) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-3">
      <h3 className="mb-3 text-xs font-medium text-gray-400 uppercase tracking-wider">Pipeline</h3>
      <div className="space-y-0.5">
        {agents.map((agent, i) => {
          const agentSubsteps = substeps[agent.name] || []
          const isRunning = agent.status === 'running'
          return (
            <div key={agent.name}>
              <div className="flex items-center gap-2 rounded p-1.5">
                {STATUS_ICON[agent.status]}
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium truncate">{agent.name.replace('_Agent', '')}</p>
                  {agent.duration_s != null && (
                    <p className="text-[10px] text-gray-500">{agent.duration_s.toFixed(1)}s</p>
                  )}
                </div>
                <span className="text-[10px] text-gray-600">{i + 1}</span>
              </div>
              {/* Substeps (shown when agent is running or just completed) */}
              {agentSubsteps.length > 0 && (isRunning || agent.status === 'completed') && (
                <div className="ml-6 mt-0.5 mb-1 space-y-0.5">
                  {agentSubsteps.map((sub) => (
                    <div key={sub.name} className="flex items-center gap-1.5">
                      {SUBSTEP_ICON[sub.status]}
                      <span className={`text-[10px] ${sub.status === 'completed' ? 'text-gray-500' : 'text-gray-300'}`}>
                        {sub.name}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
