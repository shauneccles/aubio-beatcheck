# Aubio BeatCheck

A comprehensive validation and benchmarking tool for testing aubio audio analysis functions. Designed for **closed-loop optimization** of aubio parameters and performance tracking.

## Features

- 🎯 **Standard Test Suites** - Comprehensive pre-built test signals for tempo, onset, pitch, and complex analysis
- 📊 **Evaluation Metrics** - Precision, recall, F-measure, and timing accuracy (MAE)
- 📈 **Waveform Visualization** - Interactive plots comparing ground truth vs detected events
- 💾 **JSON Artifacts** - Machine-readable outputs for AI-driven optimization
- ⚡ **Performance Metrics** - Frame processing times (mean, P95, P99)
- 🌐 **Web UI + CLI** - Choose between interactive web interface or scriptable CLI

## Installation

```bash
# Clone the repository
git clone https://github.com/shauneccles/aubio-beatcheck.git
cd aubio-beatcheck

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### CLI Mode (for AI Agent Integration)

Run analysis and generate JSON/PNG artifacts:

```bash
# Run tempo suite with 10-second signals
aubio-beatcheck run --suite tempo --duration 10 --output ./results

# Run all suites
aubio-beatcheck run --suite all --duration 15 --output ./results

# Output structure:
# ./results/
# ├── test_input.json          # Test configuration
# ├── ground_truth.json        # Expected events
# ├── analysis_results.json    # Detected events + metrics
# ├── evaluation.json          # Precision/recall/F1
# └── plots/
#     └── *.png                # Waveform visualizations
```

### Web UI Mode

Start the web server for interactive analysis:

```bash
# Start backend
aubio-beatcheck web --port 8000

# Start frontend (in another terminal)
cd web && npm run dev

# Open http://localhost:5173
```

## Test Suites

| Suite | Description | Tests |
|-------|-------------|-------|
| `tempo` | Beat tracking & BPM detection | 60-180 BPM, timing jitter, click durations |
| `onset` | Transient detection | Attack types (impulse, sharp, medium, slow) |
| `pitch` | Fundamental frequency | Chromatic scales, intervals, waveforms |
| `rhythmic` | Complex patterns | Syncopation, polyrhythms |
| `complex` | Combined signals | Beats + melody + noise at various SNR |
| `all` | All of the above | Full validation suite |

## API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/suites` | List available test suites |
| `POST` | `/api/suites/{id}/run` | Start analysis |
| `GET` | `/api/results/{suite_id}` | Get analysis results |
| `GET` | `/api/results/{suite_id}/{signal}/plot` | Get waveform plot (PNG) |

### JSON Artifact Formats

See [docs/API.md](docs/API.md) for detailed schema documentation.

## Development

```bash
# Install dev dependencies
uv sync --all-extras --dev

# Run tests
uv run pytest tests/ -v

# Lint & format
uv run ruff check .
uv run ruff format .
```

## Architecture

```
aubio-beatcheck/
├── aubio_beatcheck/          # Core package
│   ├── cli.py                # CLI entry point
│   ├── core/                 # Analysis logic
│   │   ├── analyzers.py      # Aubio wrappers
│   │   ├── evaluation.py     # Metrics calculation
│   │   ├── ground_truth.py   # Pydantic schemas
│   │   └── thebeat_gen.py    # Signal generation
│   └── suites/               # Test suite definitions
├── web_api/                  # Litestar REST API
│   ├── main.py
│   ├── routers.py
│   └── plotting.py
├── web/                      # React frontend
└── tests/                    # pytest tests
```

## References

- [aubio](https://aubio.org/) - Audio analysis library
- [thebeat](https://github.com/Nian-Jingqing/thebeat) - Research-validated signal generation
- [Litestar](https://litestar.dev/) - Python ASGI framework

## License

GPL-3.0
