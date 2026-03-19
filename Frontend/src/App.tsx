import { BrowserRouter, Routes, Route } from 'react-router-dom'
import HomePage from './pages/HomePage'
import PipelinePage from './pages/PipelinePage'
import ReportsPage from './pages/ReportsPage'

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-950 text-gray-100">
        <header className="border-b border-gray-800 px-6 py-4">
          <h1 className="text-xl font-semibold tracking-tight">
            Credit Risk Modeling Platform
          </h1>
          <p className="text-sm text-gray-400">PD / LGD / EAD / Expected Loss Pipeline</p>
        </header>
        <main className="mx-auto max-w-7xl px-6 py-8">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/pipeline/:runId" element={<PipelinePage />} />
            <Route path="/reports/:runId" element={<ReportsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App
