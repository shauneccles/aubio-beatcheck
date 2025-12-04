import { useState, useEffect } from 'react'
import './App.css'
import SuiteSelector from './components/SuiteSelector'
import AnalysisProgress from './components/AnalysisProgress'
import ResultsView from './components/ResultsView'
import { LayoutDashboard, Music, Activity } from 'lucide-react'

function App() {
  const [currentView, setCurrentView] = useState('selector') // selector, analysis, results
  const [selectedSuite, setSelectedSuite] = useState(null)
  const [analysisConfig, setAnalysisConfig] = useState({ duration: 10.0 })
  const [results, setResults] = useState(null)
  const [analysisTask, setAnalysisTask] = useState(null)

  const startAnalysis = async (suiteId, config) => {
    setSelectedSuite(suiteId)
    setAnalysisConfig(config)
    setCurrentView('analysis')

    try {
      const response = await fetch(`http://localhost:8000/api/analyze/${suiteId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      })
      const data = await response.json()
      setAnalysisTask(data)
    } catch (error) {
      console.error("Failed to start analysis:", error)
      // Handle error
    }
  }

  const handleAnalysisComplete = (data) => {
    setResults(data)
    setCurrentView('results')
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="logo">
          <Activity className="icon" />
          <h1>Aubio BeatCheck</h1>
        </div>
        <nav>
          <button onClick={() => setCurrentView('selector')} className={currentView === 'selector' ? 'active' : ''}>
            <LayoutDashboard size={18} /> Suites
          </button>
          <button onClick={() => setCurrentView('results')} disabled={!results} className={currentView === 'results' ? 'active' : ''}>
            <Music size={18} /> Results
          </button>
        </nav>
      </header>

      <main className="app-content">
        {currentView === 'selector' && (
          <SuiteSelector onSelect={startAnalysis} />
        )}
        {currentView === 'analysis' && (
          <AnalysisProgress
            suiteId={selectedSuite}
            task={analysisTask}
            onComplete={handleAnalysisComplete}
          />
        )}
        {currentView === 'results' && (
          <ResultsView results={results} suiteId={selectedSuite} />
        )}
      </main>
    </div>
  )
}

export default App
