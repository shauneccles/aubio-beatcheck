import { useState } from 'react'
import { CheckCircle, XCircle, Activity } from 'lucide-react'

export default function ResultsView({ results, suiteId }) {
    const [expandedPlots, setExpandedPlots] = useState(new Set())

    if (!results) return <div>No results available</div>

    const successCount = results.filter(r => r.status === 'completed').length
    const totalCount = results.length

    const togglePlot = (signalName) => {
        const newExpanded = new Set(expandedPlots)
        if (newExpanded.has(signalName)) {
            newExpanded.delete(signalName)
        } else {
            newExpanded.add(signalName)
        }
        setExpandedPlots(newExpanded)
    }

    return (
        <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
                <h2>Results: {suiteId}</h2>
                <div className="card" style={{ padding: '0.5rem 1rem' }}>
                    <span className="text-success" style={{ fontWeight: 'bold' }}>{successCount}/{totalCount} Passed</span>
                </div>
            </div>

            <div className="card">
                <table>
                    <thead>
                        <tr>
                            <th>Signal</th>
                            <th>Status</th>
                            <th>Metrics</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {results.map((result, idx) => (
                            <>
                                <tr key={idx}>
                                    <td>{result.signal_name}</td>
                                    <td>
                                        {result.status === 'completed' ? (
                                            <span className="text-success" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                                <CheckCircle size={16} /> Completed
                                            </span>
                                        ) : (
                                            <span className="text-error" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                                <XCircle size={16} /> Failed
                                            </span>
                                        )}
                                    </td>
                                    <td>
                                        <div style={{ fontSize: '0.9rem' }}>
                                            {result.metrics.tempo_bpm && (
                                                <div><strong>BPM:</strong> {result.metrics.tempo_bpm.toFixed(1)}</div>
                                            )}
                                            {result.metrics.beat_count !== undefined && (
                                                <div><strong>Beats:</strong> {result.metrics.beat_count}</div>
                                            )}
                                            {result.metrics.onset_count !== undefined && (
                                                <div><strong>Onsets:</strong> {result.metrics.onset_count}</div>
                                            )}
                                            {result.metrics.pitch_count !== undefined && (
                                                <div><strong>Pitches:</strong> {result.metrics.pitch_count}</div>
                                            )}

                                            {result.metrics.evaluation && (
                                                <div style={{ marginTop: '0.5rem', borderTop: '1px solid var(--border)', paddingTop: '0.5rem' }}>
                                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', fontSize: '0.8rem' }}>
                                                        <div><strong>Prec:</strong> {result.metrics.evaluation.precision.toFixed(2)}</div>
                                                        <div><strong>Rec:</strong> {result.metrics.evaluation.recall.toFixed(2)}</div>
                                                        <div><strong>F1:</strong> {result.metrics.evaluation.f_measure.toFixed(2)}</div>
                                                        <div><strong>MAE:</strong> {result.metrics.evaluation.mae_ms.toFixed(1)}ms</div>
                                                    </div>
                                                    {(result.metrics.evaluation.false_positives.length > 0 || result.metrics.evaluation.false_negatives.length > 0) && (
                                                        <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--error)' }}>
                                                            {result.metrics.evaluation.false_positives.length > 0 && (
                                                                <div>False Pos: {result.metrics.evaluation.false_positives.length}</div>
                                                            )}
                                                            {result.metrics.evaluation.false_negatives.length > 0 && (
                                                                <div>Missed: {result.metrics.evaluation.false_negatives.length}</div>
                                                            )}
                                                        </div>
                                                    )}
                                                </div>
                                            )}

                                            {result.metrics.stats && (
                                                <div style={{ marginTop: '0.5rem', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                                                    <div>Mean: {result.metrics.stats.mean_us.toFixed(1)}µs</div>
                                                    <div>P95: {result.metrics.stats.p95_us.toFixed(1)}µs</div>
                                                </div>
                                            )}
                                        </div>
                                    </td>
                                    <td>
                                        <button
                                            className="btn btn-secondary"
                                            onClick={() => togglePlot(result.signal_name)}
                                            style={{ padding: '0.25rem 0.5rem', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.25rem' }}
                                        >
                                            <Activity size={14} />
                                            {expandedPlots.has(result.signal_name) ? 'Hide Plot' : 'Show Plot'}
                                        </button>
                                    </td>
                                </tr>
                                {expandedPlots.has(result.signal_name) && (
                                    <tr>
                                        <td colSpan="4" style={{ padding: '1rem', background: 'var(--bg-secondary)' }}>
                                            <img
                                                src={`http://localhost:8000/api/results/${suiteId}/${result.signal_name}/plot`}
                                                alt={`Analysis plot for ${result.signal_name}`}
                                                style={{ width: '100%', borderRadius: '4px', border: '1px solid var(--border)' }}
                                            />
                                        </td>
                                    </tr>
                                )}
                            </>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    )
}
