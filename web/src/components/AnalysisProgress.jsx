import { useState, useEffect } from 'react'
import { Loader2 } from 'lucide-react'

export default function AnalysisProgress({ suiteId, task, onComplete }) {
    const [status, setStatus] = useState('Initializing...')
    const [progress, setProgress] = useState(0)

    useEffect(() => {
        if (!suiteId) return

        // Poll for results
        // In a real app, we'd use WebSocket or SSE or the background task status
        // For this demo, we'll poll the results endpoint until we get data

        const interval = setInterval(() => {
            fetch(`http://localhost:8000/api/results/${suiteId}`)
                .then(res => res.json())
                .then(data => {
                    if (data && data.length > 0) {
                        // Check if all are completed (simulated)
                        // For now, just assume if we get data, it's done
                        clearInterval(interval)
                        onComplete(data)
                    } else {
                        // Simulate progress
                        setProgress(prev => Math.min(prev + 10, 90))
                        setStatus('Analyzing signals...')
                    }
                })
                .catch(err => console.error(err))
        }, 1000)

        return () => clearInterval(interval)
    }, [suiteId, onComplete])

    return (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '400px' }}>
            <Loader2 className="spin" size={48} color="var(--primary)" style={{ animation: 'spin 1s linear infinite' }} />
            <h2 style={{ marginTop: '1rem' }}>Running Analysis: {suiteId}</h2>
            <p style={{ color: 'var(--text-muted)' }}>{status}</p>

            <div style={{ width: '300px', height: '4px', background: 'var(--bg-card)', marginTop: '1rem', borderRadius: '2px', overflow: 'hidden' }}>
                <div style={{ width: `${progress}%`, height: '100%', background: 'var(--primary)', transition: 'width 0.3s' }}></div>
            </div>

            <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
        </div>
    )
}
