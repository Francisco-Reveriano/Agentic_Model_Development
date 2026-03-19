import { useEffect, useMemo, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { TableEntry } from '../hooks/usePipelineState'

interface LogEntry {
  agent: string
  content: string
  type: string
  ts: number
  tableData?: TableEntry
}

interface Props {
  logs: LogEntry[]
}

interface MergedBlock {
  agent: string
  type: 'text' | 'reasoning' | 'table' | string
  content: string
  tableData?: TableEntry
}

function InlineTable({ data }: { data: TableEntry }) {
  return (
    <div className="my-2 rounded-md border border-gray-700/50 overflow-hidden">
      <div className="bg-gray-800/80 px-3 py-1.5 flex items-center justify-between">
        <span className="text-[10px] font-medium text-gray-400">{data.table_name}</span>
        <span className="text-[9px] text-gray-600">{data.rows.length} rows</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-[11px] font-mono border-collapse">
          <thead>
            <tr className="bg-gray-800/40">
              {data.columns.map((col) => (
                <th key={col} className="px-2 py-1 text-left text-[10px] font-semibold text-gray-500 border-b border-gray-700/50 whitespace-nowrap">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row, i) => (
              <tr key={i} className="border-b border-gray-800/30 hover:bg-gray-800/20">
                {(row as unknown[]).map((cell, j) => (
                  <td key={j} className="px-2 py-0.5 text-gray-300 whitespace-nowrap">
                    {cell == null ? <span className="text-gray-700 italic">null</span> : String(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// Tailwind classes for dark-themed markdown rendering
const mdStyles = [
  'max-w-none text-sm leading-relaxed',
  '[&_h1]:text-base [&_h1]:font-bold [&_h1]:text-blue-300 [&_h1]:mt-4 [&_h1]:mb-2',
  '[&_h2]:text-sm [&_h2]:font-semibold [&_h2]:text-blue-300/90 [&_h2]:mt-3 [&_h2]:mb-1.5 [&_h2]:border-b [&_h2]:border-gray-800 [&_h2]:pb-1',
  '[&_h3]:text-sm [&_h3]:font-medium [&_h3]:text-gray-300 [&_h3]:mt-2 [&_h3]:mb-1',
  '[&_strong]:text-amber-300 [&_strong]:font-semibold',
  '[&_em]:text-gray-400',
  '[&_code]:bg-gray-800 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:rounded [&_code]:text-xs [&_code]:text-emerald-300 [&_code]:font-mono',
  '[&_pre]:bg-gray-800 [&_pre]:rounded-md [&_pre]:p-3 [&_pre]:overflow-x-auto [&_pre]:text-xs [&_pre]:my-2',
  '[&_pre_code]:bg-transparent [&_pre_code]:p-0',
  '[&_ul]:list-disc [&_ul]:pl-5 [&_ul]:space-y-0.5 [&_ul]:my-1.5',
  '[&_ol]:list-decimal [&_ol]:pl-5 [&_ol]:space-y-0.5 [&_ol]:my-1.5',
  '[&_li]:text-gray-300 [&_li]:text-sm',
  '[&_p]:mb-2 [&_p]:text-gray-200',
  '[&_table]:text-xs [&_table]:border-collapse [&_table]:w-full [&_table]:my-2',
  '[&_th]:bg-gray-800 [&_th]:px-2 [&_th]:py-1 [&_th]:text-left [&_th]:text-gray-400 [&_th]:border [&_th]:border-gray-700 [&_th]:font-medium',
  '[&_td]:px-2 [&_td]:py-1 [&_td]:border [&_td]:border-gray-700/50 [&_td]:text-gray-300',
  '[&_tr:nth-child(even)]:bg-gray-800/20',
  '[&_blockquote]:border-l-2 [&_blockquote]:border-blue-500/40 [&_blockquote]:pl-3 [&_blockquote]:text-gray-400 [&_blockquote]:italic [&_blockquote]:my-2',
  '[&_hr]:border-gray-800 [&_hr]:my-3',
  '[&_a]:text-blue-400 [&_a]:underline',
].join(' ')

export default function StreamingLog({ logs }: Props) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs.length])

  const blocks = useMemo<MergedBlock[]>(() => {
    const result: MergedBlock[] = []
    for (const log of logs) {
      if (log.type === 'table') {
        result.push({ agent: log.agent, type: 'table', content: log.content, tableData: log.tableData })
        continue
      }
      const last = result[result.length - 1]
      if (
        last &&
        last.type !== 'table' &&
        last.agent === log.agent &&
        (last.type === log.type || (last.type === 'text' && log.type === 'reasoning') || (last.type === 'reasoning' && log.type === 'text'))
      ) {
        last.content += log.content
      } else {
        result.push({ agent: log.agent, type: log.type, content: log.content })
      }
    }
    return result
  }, [logs])

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4 flex flex-col h-full">
      <h3 className="mb-3 text-sm font-medium text-gray-400">Agent Output</h3>
      <div className="flex-1 overflow-y-auto rounded bg-gray-950 p-4 text-sm leading-relaxed space-y-3 min-h-0">
        {blocks.length === 0 && (
          <p className="text-gray-600 italic">Waiting for pipeline output...</p>
        )}
        {blocks.map((block, i) => {
          if (block.type === 'table' && block.tableData) {
            return (
              <div key={i}>
                <span className="text-[10px] text-blue-400/60 font-mono">{block.agent}</span>
                <InlineTable data={block.tableData} />
              </div>
            )
          }
          return (
            <div key={i} className="group">
              <div className="flex items-start gap-2">
                <span className="text-[10px] text-blue-400/60 mt-0.5 flex-shrink-0 font-mono">
                  {block.agent}
                </span>
                <div className={`flex-1 min-w-0 ${mdStyles} ${
                  block.type === 'reasoning' ? 'opacity-60 italic' : ''
                }`}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {block.content}
                  </ReactMarkdown>
                </div>
              </div>
            </div>
          )
        })}
        <div ref={endRef} />
      </div>
    </div>
  )
}
