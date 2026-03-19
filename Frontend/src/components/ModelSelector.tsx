import { useState } from 'react'

interface Props {
  onStart: (models: string[]) => void
  loading: boolean
}

const OPTIONS = [
  { label: 'PD Only', value: ['PD'], desc: 'Probability of Default model' },
  { label: 'LGD Only', value: ['LGD'], desc: 'Loss Given Default model' },
  { label: 'EAD Only', value: ['EAD'], desc: 'Exposure at Default model' },
  { label: 'All Models + EL', value: ['PD', 'LGD', 'EAD', 'EL'], desc: 'Full pipeline with Expected Loss' },
]

export default function ModelSelector({ onStart, loading }: Props) {
  const [selected, setSelected] = useState(3) // default: All Models

  return (
    <section className="rounded-lg border border-gray-800 bg-gray-900 p-6">
      <h2 className="mb-4 text-lg font-medium">Select Models</h2>
      <div className="space-y-3">
        {OPTIONS.map((opt, i) => (
          <label
            key={i}
            className={`flex cursor-pointer items-center gap-3 rounded-md border p-4 transition-colors ${
              selected === i
                ? 'border-blue-500 bg-blue-500/10'
                : 'border-gray-700 hover:border-gray-600'
            }`}
          >
            <input
              type="radio"
              name="model"
              checked={selected === i}
              onChange={() => setSelected(i)}
              className="accent-blue-500"
            />
            <div>
              <span className="font-medium">{opt.label}</span>
              <p className="text-sm text-gray-400">{opt.desc}</p>
            </div>
          </label>
        ))}
      </div>
      <button
        onClick={() => onStart(OPTIONS[selected].value)}
        disabled={loading}
        className="mt-6 w-full rounded-md bg-blue-600 py-3 font-medium transition-colors hover:bg-blue-500 disabled:opacity-50"
      >
        {loading ? 'Starting...' : 'Start Pipeline'}
      </button>
    </section>
  )
}
