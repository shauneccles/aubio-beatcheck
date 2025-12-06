# Contributing to Aubio BeatCheck

## Code Style

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check . --fix

# Format code
uv run ruff format .
```

## Type Annotations

All functions must have complete type annotations:

```python
# ✅ Good
def analyze_signal(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float = 0.3,
) -> list[float]:
    """Analyze audio signal for onsets."""
    ...

# ❌ Bad
def analyze_signal(audio, sample_rate, threshold=0.3):
    ...
```

## Docstrings

Use Google-style docstrings:

```python
def evaluate_events(
    detected: list[float],
    ground_truth: list[float],
    tolerance_ms: float = 50.0,
) -> EvaluationMetrics:
    """Evaluate detected events against ground truth.

    Args:
        detected: List of detected event timestamps in seconds.
        ground_truth: List of ground truth event timestamps in seconds.
        tolerance_ms: Maximum timing error for a match in milliseconds.

    Returns:
        EvaluationMetrics containing precision, recall, F-measure, and MAE.

    Raises:
        ValueError: If tolerance_ms is negative.
    """
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature (`git checkout -b feature/my-feature`)
3. **Make changes** following the style guidelines above
4. **Run tests** (`uv run pytest tests/ -v`)
5. **Run linting** (`uv run ruff check . && uv run ruff format --check .`)
6. **Submit PR** with a clear description

## Adding Test Suites

To add a new test suite:

1. Add signal generator in `aubio_beatcheck/core/thebeat_gen.py`
2. Create suite function in `aubio_beatcheck/suites/standard.py`
3. Register in `StandardSuites.get_suite()`
4. Add corresponding API route if needed

## Reporting Issues

When reporting bugs, please include:

- Python version (`python --version`)
- aubio version (`python -c "import aubio; print(aubio.version)"`)
- Full error traceback
- Minimal reproduction steps
