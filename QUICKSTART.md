# Aubio BeatCheck - Quick Start Guide

> **Note:** This project has been refactored from a pytest test suite into a standalone Textual TUI application.
> This guide covers the new application usage.

## Installation

Install aubio-beatcheck with all dependencies:

```bash
# Using uv (recommended)
uv sync

# Using pip
pip install -e .
```

## 5-Minute Tutorial

### 1. Launch the Application

```bash
# Interactive TUI mode
aubio-beatcheck

# Or use the Python module
python main.py
```

### 2. Select a Test Suite

The main menu presents standard test suites:

- **Tempo/Beat Tracking** - Tests BPM detection (60-180 BPM range)
- **Onset Detection** - Tests transient detection with various attacks
- **Pitch Detection** - Tests pitch tracking across musical range
- **Rhythmic Patterns** - Tests complex rhythmic patterns
- **Complex Signals** - Tests combined beats+melody+noise

Select a suite with arrow keys and press Enter.

### 3. Configure and Run

- Choose signal duration (5, 10, 15, or 30 seconds)
- Press "Run Analysis" to start
- Watch real-time progress with live metrics

### 4. View Results

Results screen shows:
- Summary statistics (success rate, category breakdown)
- Detailed per-signal results (tempo, onsets, pitches)
- Performance metrics (frame processing times)

### 5. Export (Coming Soon)

Export results as JSON, markdown, or PNG visualizations.

## Using as a Library

### Generate Test Signals

```python
from aubio_beatcheck.core.signals import SignalGenerator

# Generate a 120 BPM click track
audio, signal_def = SignalGenerator.click_track(bpm=120, duration=10.0)

print(f"Generated {len(audio)} samples")
print(f"Ground truth beats: {len(signal_def.ground_truth.beats)}")
print(f"Actual BPM: {signal_def.metadata.bpm:.1f}")
```

### Analyze with Aubio

```python
from aubio_beatcheck.core.analyzers import TempoAnalyzer, AnalyzerConfig
from aubio_beatcheck.core.signals import SignalGenerator

# Generate signal
audio, signal_def = SignalGenerator.click_track(bpm=140, duration=30.0)

# Configure analyzer
config = AnalyzerConfig(fft_size=2048, hop_size=512, sample_rate=44100)
tempo_analyzer = TempoAnalyzer(config)

# Run analysis
detected_beats, detected_bpm = tempo_analyzer.analyze(audio)

print(f"Expected: {signal_def.metadata.bpm:.1f} BPM")
print(f"Detected: {detected_bpm:.1f} BPM")
print(f"Found {len(detected_beats)} beats")
```

### Use Standard Test Suites

```python
from aubio_beatcheck.suites.standard import StandardSuites

# Get tempo test suite
tempo_signals = StandardSuites.tempo_suite(duration=10.0)
print(f"Tempo suite has {len(tempo_signals)} signals")

# Get all suites
all_suites = StandardSuites.all_suites(duration=10.0)
for suite_name, signals in all_suites.items():
    print(f"{suite_name}: {len(signals)} signals")
```

## Common Use Cases

### CLI Mode - Run Specific Suite

```bash
# Run tempo suite with 15-second signals
aubio-beatcheck --suite tempo --duration 15

# Run all suites
aubio-beatcheck --suite all --duration 10

# Analyze custom audio (coming soon)
aubio-beatcheck --audio my_track.wav --ground-truth annotations.json
```

### Programmatic Analysis

```python
from aubio_beatcheck.core.analyzers import OnsetAnalyzer, PitchAnalyzer
from aubio_beatcheck.core.signals import SignalGenerator

# Generate onset signal
audio, signal_def = SignalGenerator.onset_signal(
    attack_type="sharp",
    interval_ms=500.0,
    duration=5.0,
)

# Analyze onsets
onset_analyzer = OnsetAnalyzer()
detected_onsets = onset_analyzer.analyze(audio)
expected_onsets = [o.time for o in signal_def.ground_truth.onsets]

print(f"Expected: {len(expected_onsets)} onsets")
print(f"Detected: {len(detected_onsets)} onsets")
```

### Performance Metrics

```python
from aubio_beatcheck.core.analyzers import TempoAnalyzer

analyzer = TempoAnalyzer()
beats, bpm = analyzer.analyze(audio)

# Check performance stats
print(f"Mean frame time: {analyzer.stats.mean_us:.1f} μs")
print(f"P95 frame time: {analyzer.stats.p95_us:.1f} μs")
print(f"Max frame time: {analyzer.stats.max_us:.1f} μs")
```

## Tips & Tricks

### 1. Reproducible Testing

Always set `rng_seed` for reproducible jitter:

```python
audio, signal_def = SignalGenerator.click_track(
    bpm=120,
    duration=10.0,
    add_timing_jitter=True,
    jitter_std_ms=10.0,
    rng_seed=42,  # Same seed = same jitter
)
```

### 2. Custom Test Suites

Create your own test suite:

```python
from aubio_beatcheck.suites.standard import TestSignal
from aubio_beatcheck.core.signals import SignalGenerator

def my_custom_suite():
    signals = []
    
    # Add custom BPM tests
    for bpm in [75, 133, 155]:
        audio, signal_def = SignalGenerator.click_track(bpm=bpm, duration=10.0)
        signals.append(TestSignal(
            name=f"click_{bpm}bpm",
            description=f"{bpm} BPM test",
            audio=audio,
            signal_def=signal_def,
            category="tempo"
        ))
    
    return signals
```

### 3. Batch Analysis

```python
from aubio_beatcheck.suites.standard import StandardSuites
from aubio_beatcheck.core.analyzers import TempoAnalyzer

# Get signals
signals = StandardSuites.tempo_suite(duration=10.0)

# Analyze all
analyzer = TempoAnalyzer()
results = []

for signal in signals:
    beats, bpm = analyzer.analyze(signal.audio)
    results.append({
        "name": signal.name,
        "expected_bpm": signal.signal_def.metadata.bpm,
        "detected_bpm": bpm,
        "error": abs(bpm - signal.signal_def.metadata.bpm)
    })
    analyzer.reset()  # Reset for next signal

# Print summary
for r in results:
    print(f"{r['name']}: {r['detected_bpm']:.1f} BPM (error: {r['error']:.1f})")
```

## Troubleshooting

### Import Error: "No module named 'textual'"

```bash
# Make sure all dependencies are installed
uv sync

# Or with pip
pip install textual
```

### Import Error: "No module named 'ground_truth_schema'"

The application temporarily imports from the `src/` directory. This will be resolved when we migrate all modules to `aubio_beatcheck/core/`.

```python
# Workaround: Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
```

### TUI Not Launching

Make sure you're using Python 3.12+ and have a compatible terminal:

```bash
# Check Python version
python --version  # Should be 3.12 or higher

# Try running directly
python -m aubio_beatcheck.cli
```

## Next Steps

1. **Explore the TUI** - Run `aubio-beatcheck` and try different suites
2. **Read the full README** - See `README.md` for architecture details
3. **Check legacy docs** - See `THEBEAT_README.md` and `IMPLEMENTATION_SUMMARY.md` for thebeat integration details
4. **Contribute** - Add custom test suites or improve the UI

## Quick Reference

| Component | Purpose | Example |
|-----------|---------|---------|
| `SignalGenerator` | Generate test signals | `SignalGenerator.click_track(bpm=120)` |
| `TempoAnalyzer` | Tempo/beat analysis | `analyzer.analyze(audio)` |
| `OnsetAnalyzer` | Onset detection | `analyzer.analyze(audio)` |
| `PitchAnalyzer` | Pitch detection | `analyzer.analyze(audio)` |
| `StandardSuites` | Pre-built test suites | `StandardSuites.tempo_suite()` |
| `aubio-beatcheck` | Launch TUI | `aubio-beatcheck` (in terminal) |
| CLI mode | Non-interactive | `aubio-beatcheck --suite tempo` |

## Migration from Old Pytest Suite

If you're familiar with the old test suite:

**Old (pytest):**
```python
from tests.test_multifft.thebeat_generator import generate_click_track
audio, signal_def = generate_click_track(bpm=120, duration=10.0)
```

**New (application):**
```python
from aubio_beatcheck.core.signals import SignalGenerator
audio, signal_def = SignalGenerator.click_track(bpm=120, duration=10.0)
```

The core thebeat generation functionality is the same, just reorganized into a cleaner package structure.

Happy analyzing! 🎵
