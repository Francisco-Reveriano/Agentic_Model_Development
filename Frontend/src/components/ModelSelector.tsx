import { useState } from 'react'

interface Props {
  onStart: (models: string[]) => void
  loading: boolean
}

const OPTIONS = [
  { label: 'PD Only', value: ['PD'], desc: 'Probability of Default', color: 'bg-blue-500' },
  { label: 'LGD Only', value: ['LGD'], desc: 'Loss Given Default', color: 'bg-emerald-500' },
  { label: 'EAD Only', value: ['EAD'], desc: 'Exposure at Default', color: 'bg-amber-500' },
  { label: 'All Models + EL', value: ['PD', 'LGD', 'EAD', 'EL'], desc: 'Full pipeline with Expected Loss', color: 'bg-purple-500' },
]

export default function ModelSelector({ onStart, loading }: Props) {
  const [selected, setSelected] = useState(3)

  return (
    <section className="rounded-lg border border-gray-800 bg-gray-900/80 backdrop-blur p-5">
      <h2 className="mb-4 text-sm font-medium text-gray-400 uppercase tracking-wider">Select Models</h2>
      <div className="space-y-2">
        {OPTIONS.map((opt, i) => (
          <label
            key={i}
            className={`flex cursor-pointer items-center gap-3 rounded-lg border px-4 py-3 transition-all duration-200 ${
              selected === i
                ? 'border-blue-500/60 bg-blue-500/10 shadow-[0_0_15px_-3px_rgba(59,130,246,0.3)]'
                : 'border-gray-700/50 hover:border-gray-600 hover:bg-gray-800/50'
            }`}
          >
            <input
              type="radio"
              name="model"
              checked={selected === i}
              onChange={() => setSelected(i)}
              className="sr-only"
            />
            <div className="relative flex-shrink-0">
              <div className={`h-2.5 w-2.5 rounded-full ${opt.color} ${selected === i ? 'ring-2 ring-offset-2 ring-offset-gray-900' : 'opacity-50'} transition-all`}
                style={selected === i ? { ringColor: opt.color } : {}}
              />
              {selected === i && (
                <div className={`absolute inset-0 h-2.5 w-2.5 rounded-full ${opt.color} animate-ping opacity-40`} />
              )}
            </div>
            <div className="min-w-0">
              <span className="text-sm font-medium">{opt.label}</span>
              <p className="text-xs text-gray-500">{opt.desc}</p>
            </div>
          </label>
        ))}
      </div>
      <button
        onClick={() => onStart(OPTIONS[selected].value)}
        disabled={loading}
        className="mt-5 w-full rounded-lg bg-gradient-to-r from-blue-600 to-blue-500 py-2.5 text-sm font-semibold tracking-wide transition-all hover:from-blue-500 hover:to-blue-400 hover:shadow-[0_0_20px_-3px_rgba(59,130,246,0.5)] disabled:opacity-50 disabled:hover:shadow-none"
      >
        {loading ? 'Starting...' : 'Start Pipeline'}
      </button>
    </section>
  )
}
