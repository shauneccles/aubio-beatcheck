import { useState, useEffect } from 'react'
import { Play } from 'lucide-react'

export default function SuiteSelector({ onSelect }) {
    const [suites, setSuites] = useState([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        fetch('http://localhost:8000/api/suites')
            .then(res => res.json())
            .then(data => {
                setSuites(data)
                setLoading(false)
            })
            .catch(err => {
                console.error(err)
                setLoading(false)
            })
    }, [])

    if (loading) return <div>Loading suites...</div>

    return (
        <div>
            <h2 style={{ marginBottom: '1.5rem' }}>Select Test Suite</h2>
            <div className="grid">
                {suites.map(suite => (
                    <div key={suite.id} className="card">
                        <h3>{suite.name}</h3>
                        <p>{suite.description}</p>
                        <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'flex-end' }}>
                            <button
                                className="primary"
                                onClick={() => onSelect(suite.id, { duration: 10.0 })}
                            >
                                <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <Play size={16} /> Run Analysis
                                </span>
                            </button>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    )
}
