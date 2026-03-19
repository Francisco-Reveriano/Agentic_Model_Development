import { useEffect, useRef, useState, useCallback } from 'react'
import type { SSEEvent, SSEEventType } from '../types/events'

export type ConnectionStatus = 'connecting' | 'open' | 'closed' | 'error'

export function useSSE(url: string | null) {
  const [events, setEvents] = useState<SSEEvent[]>([])
  const [status, setStatus] = useState<ConnectionStatus>('closed')
  const sourceRef = useRef<EventSource | null>(null)

  const close = useCallback(() => {
    sourceRef.current?.close()
    sourceRef.current = null
    setStatus('closed')
  }, [])

  useEffect(() => {
    if (!url) return

    setStatus('connecting')
    const es = new EventSource(url)
    sourceRef.current = es

    es.onopen = () => setStatus('open')
    es.onerror = () => setStatus('error')

    const EVENT_TYPES: SSEEventType[] = [
      'agent_start', 'agent_log', 'agent_metric', 'agent_complete',
      'agent_error', 'pipeline_start', 'pipeline_complete', 'pipeline_error',
      'tournament_start', 'model_trained', 'phase_complete',
      'feature_consensus', 'iteration_update', 'model_pruned',
      'champion_declared', 'heartbeat', 'stream_end',
    ]

    for (const type of EVENT_TYPES) {
      es.addEventListener(type, (e: MessageEvent) => {
        if (type === 'heartbeat') return
        if (type === 'stream_end') {
          close()
          return
        }
        try {
          const data = JSON.parse(e.data)
          setEvents((prev) => [...prev, { type, data }])
        } catch {
          // ignore parse errors
        }
      })
    }

    return () => {
      es.close()
      sourceRef.current = null
    }
  }, [url, close])

  return { events, status, close }
}
