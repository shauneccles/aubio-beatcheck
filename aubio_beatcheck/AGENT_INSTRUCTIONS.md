# Aubio Optimizer Agent Instructions

## Overview

This document provides comprehensive instructions for an AI agent to use aubio-beatcheck for optimizing the aubio library. The system enables closed-loop validation and parameter optimization of aubio's audio analysis algorithms.

## Quick Reference

| Task | Command/Module |
|------|----------------|
| Run standard tests | `uv run aubio-beatcheck run --suite all` |
| Run edge case tests | Use `EdgeCaseSuites.all_edge_cases()` |
| Check for regression | Use `FeedbackLoop.run_with_feedback()` |
| Find optimal params | Use `AdaptiveTuner.tune()` |
| Analyze signal type | Use `SignalClassifier.classify()` |

---

## 1. Understanding the Test Suites

### When to Use Each Suite

| Suite | Use Case | Expected F-measure |
|-------|----------|-------------------|
| `StandardSuites.tempo_suite()` | Baseline tempo detection validation | ≥0.90 |
| `StandardSuites.onset_suite()` | Baseline onset detection validation | ≥0.85 |
| `StandardSuites.pitch_suite()` | Baseline pitch detection validation | ≥0.90 |
| `EdgeCaseSuites.tempo_edge_cases()` | Stress-test tempo at boundaries | ≥0.70 |
| `EdgeCaseSuites.onset_edge_cases()` | Stress-test onset detection | ≥0.70 |
| `EdgeCaseSuites.robustness_suite()` | Test noise/clipping resilience | ≥0.60 |

### Decision Tree: Which Suite to Run

```
Is this a new aubio build/commit?
├── YES → Run StandardSuites first (quick regression check)
│   └── If PASS → Run EdgeCaseSuites for deeper validation
│   └── If FAIL → Investigate regression before proceeding
└── NO → Are you optimizing parameters?
    ├── YES → Use AdaptiveTuner with appropriate suite
    └── NO → Are you testing a specific fix?
        ├── Tempo fix → Run tempo_edge_cases()
        ├── Onset fix → Run onset_edge_cases()
        └── Robustness fix → Run robustness_suite()
```

---

## 2. Running Validation Tests

### CLI Method (Quick Validation)

```bash
# Run all standard suites
uv run aubio-beatcheck run --suite all

# Run specific suite
uv run aubio-beatcheck run --suite tempo

# Save results to JSON
uv run aubio-beatcheck run --suite all --output results.json
```

### Programmatic Method (Full Control)

```python
from aubio_beatcheck.suites import StandardSuites, EdgeCaseSuites
from aubio_beatcheck.core import TempoAnalyzer, Evaluator

# Generate test signals
signals = StandardSuites.tempo_suite()

results = []
for signal in signals:
    analyzer = TempoAnalyzer()
    detected_beats, bpm = analyzer.analyze(signal.audio)
    
    ground_truth = [b.time for b in signal.signal_def.ground_truth.beats]
    metrics = Evaluator.evaluate_events(detected_beats, ground_truth)
    
    results.append({
        "name": signal.name,
        "f_measure": metrics.f_measure,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "mae_ms": metrics.mean_absolute_error_ms
    })
```

---

## 3. Interpreting Results

### Key Metrics

| Metric | Good | Acceptable | Poor | Meaning |
|--------|------|------------|------|---------|
| F-measure | ≥0.95 | 0.80-0.95 | <0.80 | Overall detection accuracy |
| Precision | ≥0.95 | 0.85-0.95 | <0.85 | False positive rate (low = many FP) |
| Recall | ≥0.95 | 0.85-0.95 | <0.85 | Miss rate (low = many misses) |
| MAE (ms) | <20 | 20-50 | >50 | Timing accuracy |

### Interpreting Metric Patterns

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Low precision, high recall | Too many false positives | Increase detection threshold |
| High precision, low recall | Missing events | Decrease detection threshold |
| Both low | Fundamental algorithm issue | Check FFT/hop size, try different method |
| High MAE with good F1 | Timing drift | Check hop size alignment |
| Low F1 on edge cases only | Boundary condition issues | May be acceptable |

### Expected Results by Signal Type

**Tempo Detection:**
```
Standard signals (120 BPM, clean): F ≥ 0.95
Fast tempo (200+ BPM): F ≥ 0.85
Slow tempo (<60 BPM): F ≥ 0.80
With jitter: F ≥ 0.85
Polyrhythm: F ≥ 0.70 (harder)
```

**Onset Detection:**
```
Sharp attacks: F ≥ 0.95
Medium attacks: F ≥ 0.90
Slow attacks: F ≥ 0.75
Dense onsets (50ms): F ≥ 0.80
```

**Robustness:**
```
20dB SNR: F ≥ 0.90
10dB SNR: F ≥ 0.80
5dB SNR: F ≥ 0.65
0dB SNR: F ≥ 0.50 (may fail, acceptable)
```

---

## 4. Validating Aubio Changes

### Before Making Changes

```python
from aubio_beatcheck.core import FeedbackLoop

# Establish baseline
loop = FeedbackLoop(db_path="benchmarks.db")

# Run full validation and save baseline
from aubio_beatcheck.suites import StandardSuites

signals = StandardSuites.tempo_suite() + StandardSuites.onset_suite()
baseline_results = run_analysis(signals)  # Your analysis function

# Record baseline
action = loop.run_with_feedback(
    suite="pre_change_baseline",
    results=baseline_results,
    aubio_version=get_aubio_version(),
    config=current_config
)
```

### After Making Changes

```python
# Run same tests
post_change_results = run_analysis(signals)

# Check for regression
action = loop.run_with_feedback(
    suite="post_change",
    results=post_change_results,
    aubio_version=get_aubio_version(),
    config=current_config
)

# Interpret action
if action.action == ActionType.ROLLBACK:
    print("❌ REGRESSION DETECTED")
    print(f"F-measure dropped: {action.metrics['drift']:.3f}")
    print("Recommendation: Revert changes")
    
elif action.action == ActionType.ALERT:
    print("⚠️ MINOR REGRESSION")
    print("Recommendation: Investigate before proceeding")
    
elif action.action == ActionType.UPDATE_BASELINE:
    print("✅ IMPROVEMENT DETECTED")
    print(f"F-measure improved by: {action.metrics['drift']:.3f}")
    
else:  # MAINTAIN
    print("✓ No significant change in performance")
```

### Regression Thresholds

| Threshold | Meaning | Default |
|-----------|---------|---------|
| `alert_threshold` | Trigger warning | 0.03 (3% F-measure drop) |
| `rollback_threshold` | Recommend rollback | 0.05 (5% F-measure drop) |
| `improvement_threshold` | Update baseline | 0.02 (2% F-measure gain) |

---

## 5. Parameter Optimization

### Using AdaptiveTuner

```python
from aubio_beatcheck.core import AdaptiveTuner
from aubio_beatcheck.core.benchmark_db import BenchmarkDB

db = BenchmarkDB("benchmarks.db")
tuner = AdaptiveTuner(db)

# Generate test signal
from aubio_beatcheck.core import SignalGenerator
audio, signal_def = SignalGenerator.generate_click_track(bpm=120, duration=30)
ground_truth = [b.time for b in signal_def.ground_truth.beats]

# Run adaptive tuning
best_config, best_score = tuner.tune(
    audio=audio,
    ground_truth=ground_truth,
    suite="tempo_optimization",
    signal_type="tempo",
    max_iterations=15
)

print(f"Best config: FFT={best_config.fft_size}, HOP={best_config.hop_size}")
print(f"Best score: {best_score:.3f}")
```

### Parameter Search Space

| Parameter | Options | Impact |
|-----------|---------|--------|
| `fft_size` | 1024, 2048, 4096 | Larger = better frequency resolution, worse time resolution |
| `hop_size` | 256, 512, 1024 | Smaller = better time resolution, slower processing |
| `threshold` | 0.1 - 0.6 | Higher = fewer detections, better precision |
| `method` | varies by analyzer | Different algorithms for different signal types |

### Onset Detection Methods

| Method | Best For | Characteristics |
|--------|----------|-----------------|
| `hfc` | Percussive, sharp attacks | Default, fast |
| `complex` | Tonal signals | Phase-sensitive |
| `specflux` | Complex/noisy signals | Spectral change |
| `mkl` | Soft attacks | More sensitive |
| `energy` | General purpose | Simple, robust |

### Pitch Detection Methods

| Method | Best For | Characteristics |
|--------|----------|-----------------|
| `yinfft` | Clean tonal signals | Default, accurate |
| `yin` | Very clean signals | Original YIN |
| `fcomb` | Noisy signals | More robust |
| `mcomb` | Complex spectra | Multi-comb |

---

## 6. Signal-Aware Analysis

### Classifying Unknown Audio

```python
from aubio_beatcheck.core import SignalClassifier, AnalyzerFactory

classifier = SignalClassifier(sample_rate=44100)
profile = classifier.classify(audio)

# Use profile to understand signal
print(f"Percussive: {profile.is_percussive}")
print(f"Tonal: {profile.is_tonal}")
print(f"Estimated SNR: {profile.estimated_snr:.1f} dB")
print(f"Suggested onset method: {profile.suggested_onset_method}")
print(f"Confidence: {profile.confidence:.2f}")

# Create optimized analyzer
factory = AnalyzerFactory(sample_rate=44100)
onset_analyzer = factory.create_onset_analyzer(profile=profile)
```

### Automatic Analyzer Selection

```python
# Let factory choose optimal configuration
factory = AnalyzerFactory()
tempo, onset, pitch = factory.create_all_analyzers(audio)

# Analyzers are pre-configured based on signal characteristics
beats, bpm = tempo.analyze(audio)
onsets = onset.analyze(audio)
pitches = pitch.analyze(audio)
```

---

## 7. Multi-Objective Optimization

### When to Use Different Objectives

| Objective | Use Case |
|-----------|----------|
| `balanced` | General validation |
| `precision_focused` | When false positives are costly |
| `recall_focused` | When missing events is costly |
| `timing_focused` | When timing accuracy is critical |
| `realtime_focused` | For live performance applications |

### Pareto Optimization

```python
from aubio_beatcheck.core import ParetoOptimizer, OptimizationObjectives
from aubio_beatcheck.core.grid_search import search_tempo_params

# Run grid search
results = search_tempo_params(audio, ground_truth)

# Find Pareto-optimal configurations
pareto = ParetoOptimizer()
pareto_front = pareto.find_pareto_front(results.all_results)

print(f"Found {len(pareto_front)} Pareto-optimal configurations")

# Select based on your priorities
objectives = OptimizationObjectives.precision_focused()
best = pareto.select_from_pareto(pareto_front, objectives)
print(f"Best for precision: {best.params}")
```

---

## 8. Workflow: Complete Optimization Cycle

### Step 1: Establish Baseline

```python
# Run comprehensive baseline
from aubio_beatcheck.suites import StandardSuites, EdgeCaseSuites

all_signals = (
    StandardSuites.tempo_suite() +
    StandardSuites.onset_suite() +
    EdgeCaseSuites.all_edge_cases(duration=10.0)
)

baseline = run_full_analysis(all_signals)
save_baseline(baseline, "baseline_v1.json")
```

### Step 2: Make Changes to Aubio

```
- Modify aubio source code
- Rebuild aubio library
- Install updated aubio
```

### Step 3: Validate Changes

```python
# Run same tests
post_change = run_full_analysis(all_signals)

# Compare
for suite_name in baseline.keys():
    delta = post_change[suite_name].f_measure - baseline[suite_name].f_measure
    if delta < -0.03:
        print(f"❌ REGRESSION in {suite_name}: {delta:.3f}")
    elif delta > 0.02:
        print(f"✅ IMPROVEMENT in {suite_name}: +{delta:.3f}")
    else:
        print(f"✓ Stable in {suite_name}")
```

### Step 4: Optimize Parameters (if needed)

```python
# If performance is suboptimal, optimize
tuner = AdaptiveTuner(db)

for signal_type in ["tempo", "onset"]:
    best_config, score = tuner.tune(
        audio=test_audio,
        ground_truth=truth,
        suite=f"{signal_type}_optimization",
        signal_type=signal_type
    )
    
    # Save learned preset
    from aubio_beatcheck.presets import register_learned_preset
    register_learned_preset(f"optimized_{signal_type}", best_config)
```

### Step 5: Export Results

```python
# Export optimized presets
tuner.export_optimized_presets("learned_presets.json")

# Generate trend report
loop = FeedbackLoop()
for suite in ["tempo", "onset", "pitch"]:
    trend = loop.get_trend_report(suite)
    print(f"{suite}: {trend.trend_direction} (slope: {trend.trend_slope:.4f})")
```

---

## 9. Troubleshooting

### Common Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| All F-measures near 0 | Wrong sample rate | Check `AnalyzerConfig.sample_rate` |
| Very low recall | Threshold too high | Lower detection threshold |
| Very low precision | Threshold too low | Raise detection threshold |
| Good F1, high MAE | Hop size too large | Use smaller hop size |
| Inconsistent results | Random signal generation | Use `rng_seed` for reproducibility |

### Debugging Commands

```python
# Check aubio version
import aubio
print(f"Aubio version: {aubio.version}")

# Check analyzer configuration
print(f"FFT: {analyzer.config.fft_size}")
print(f"Hop: {analyzer.config.hop_size}")
print(f"SR: {analyzer.config.sample_rate}")

# Check performance stats
print(f"Mean frame time: {analyzer.stats.mean_us:.1f} μs")
print(f"P95 frame time: {analyzer.stats.p95_us:.1f} μs")
```

---

## 10. Key API Reference

### Core Classes

| Class | Import | Purpose |
|-------|--------|---------|
| `TempoAnalyzer` | `aubio_beatcheck.core` | Beat/tempo detection |
| `OnsetAnalyzer` | `aubio_beatcheck.core` | Onset detection |
| `PitchAnalyzer` | `aubio_beatcheck.core` | Pitch detection |
| `Evaluator` | `aubio_beatcheck.core.evaluation` | Compute metrics |
| `SignalGenerator` | `aubio_beatcheck.core` | Generate test signals |

### Optimization Classes

| Class | Import | Purpose |
|-------|--------|---------|
| `AdaptiveTuner` | `aubio_beatcheck.core` | Adaptive optimization |
| `MultiObjectiveEvaluator` | `aubio_beatcheck.core` | Composite scoring |
| `ParetoOptimizer` | `aubio_beatcheck.core` | Pareto front finding |
| `AnalyzerFactory` | `aubio_beatcheck.core` | Smart analyzer creation |
| `FeedbackLoop` | `aubio_beatcheck.core` | Continuous monitoring |

### Test Suites

| Class | Import | Purpose |
|-------|--------|---------|
| `StandardSuites` | `aubio_beatcheck.suites` | Standard validation |
| `EdgeCaseSuites` | `aubio_beatcheck.suites` | Stress testing |

---

## Summary: Agent Decision Framework

```
1. ALWAYS run StandardSuites first for quick regression check
2. If standard tests PASS, run EdgeCaseSuites for deeper validation
3. If any tests FAIL:
   a. Check if failure is in expected difficult cases (acceptable)
   b. Use SignalClassifier to understand signal characteristics
   c. Use AnalyzerFactory to get optimized configuration
   d. Use AdaptiveTuner to find better parameters
4. ALWAYS save results to benchmark database for trend tracking
5. Use FeedbackLoop to get automatic recommendations
6. Export learned presets for production use
```
