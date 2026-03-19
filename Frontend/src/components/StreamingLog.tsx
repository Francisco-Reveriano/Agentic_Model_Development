import { useEffect, useRef } from 'react'

interface LogEntry {
  agent: string
  content: string
  type: string
  ts: number
}

interface Props {
  logs: LogEntry[]
}

export default function StreamingLog({ logs }: Props) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs.length])

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <h3 className="mb-3 text-sm font-medium text-gray-400">Agent Output</h3>
      <div className="h-[500px] overflow-y-auto rounded bg-gray-950 p-3 font-mono text-xs leading-relaxed">
        {logs.length === 0 && (
          <p className="text-gray-600">Waiting for pipeline output...</p>
        )}
        {logs.map((log, i) => (
          <div key={i} className="mb-1">
            <span className="text-blue-400">[{log.agent}]</span>{' '}
            <span
              className={
                log.type === 'tool_call'
                  ? 'text-yellow-400'
                  : log.type === 'tool_result'
                  ? 'text-green-400'
                  : 'text-gray-300'
              }
            >
              {log.content}
            </span>
          </div>
        ))}
        <div ref={endRef} />
      </div>
    </div>
  )
}
