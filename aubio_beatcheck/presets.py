"""Configuration Presets for Aubio Analysis.

Pre-defined analyzer configurations optimized for different use cases.
Supports both static presets and dynamically learned configurations.
"""

from pathlib import Path
from typing import Any

from aubio_beatcheck.core.analyzers import AnalyzerConfig

# Pre-defined static configuration presets
PRESETS: dict[str, AnalyzerConfig] = {
    # Real-time processing with balanced accuracy
    "real_time": AnalyzerConfig(
        fft_size=2048,
        hop_size=512,
        sample_rate=44100,
    ),
    # High accuracy for offline analysis
    "high_accuracy": AnalyzerConfig(
        fft_size=4096,
        hop_size=512,
        sample_rate=44100,
    ),
    # Low latency for live performance
    "low_latency": AnalyzerConfig(
        fft_size=1024,
        hop_size=256,
        sample_rate=44100,
    ),
    # Fast processing for batch analysis
    "fast": AnalyzerConfig(
        fft_size=1024,
        hop_size=512,
        sample_rate=44100,
    ),
    # Maximum resolution for research
    "research": AnalyzerConfig(
        fft_size=8192,
        hop_size=512,
        sample_rate=48000,
    ),
}

# Dynamic/learned presets registry
_LEARNED_PRESETS: dict[str, AnalyzerConfig] = {}


def get_preset(name: str, prefer_learned: bool = True) -> AnalyzerConfig:
    """Get a configuration preset by name.

    Args:
        name: Preset name (static or learned).
        prefer_learned: If True, check learned presets first.

    Returns:
        AnalyzerConfig with the preset parameters.

    Raises:
        KeyError: If preset name is not found.
    """
    # Check learned presets first if preferred
    if prefer_learned and name in _LEARNED_PRESETS:
        return _LEARNED_PRESETS[name]

    # Check static presets
    if name in PRESETS:
        return PRESETS[name]

    # Check learned presets as fallback
    if name in _LEARNED_PRESETS:
        return _LEARNED_PRESETS[name]

    available_static = ", ".join(PRESETS.keys())
    available_learned = ", ".join(_LEARNED_PRESETS.keys()) if _LEARNED_PRESETS else "none"
    raise KeyError(
        f"Unknown preset '{name}'. Static: {available_static}. Learned: {available_learned}"
    )


def list_presets() -> dict[str, dict[str, Any]]:
    """List all available presets with their parameters.

    Returns:
        Dictionary mapping preset names to their parameters.
    """
    result = {}

    # Static presets
    for name, config in PRESETS.items():
        result[name] = {
            "fft_size": config.fft_size,
            "hop_size": config.hop_size,
            "sample_rate": config.sample_rate,
            "type": "static",
        }

    # Learned presets
    for name, config in _LEARNED_PRESETS.items():
        result[f"learned:{name}"] = {
            "fft_size": config.fft_size,
            "hop_size": config.hop_size,
            "sample_rate": config.sample_rate,
            "type": "learned",
        }

    return result


def register_learned_preset(name: str, config: AnalyzerConfig) -> None:
    """Register a learned preset.

    Args:
        name: Preset name (will be prefixed if not already).
        config: Analyzer configuration to register.
    """
    _LEARNED_PRESETS[name] = config


def unregister_learned_preset(name: str) -> bool:
    """Remove a learned preset.

    Args:
        name: Preset name to remove.

    Returns:
        True if preset was removed, False if not found.
    """
    if name in _LEARNED_PRESETS:
        del _LEARNED_PRESETS[name]
        return True
    return False


def clear_learned_presets() -> int:
    """Clear all learned presets.

    Returns:
        Number of presets cleared.
    """
    count = len(_LEARNED_PRESETS)
    _LEARNED_PRESETS.clear()
    return count


def save_learned_presets(path: Path | str) -> int:
    """Save learned presets to JSON file.

    Args:
        path: Output file path.

    Returns:
        Number of presets saved.
    """
    import json

    path = Path(path)
    data = {
        name: {
            "fft_size": config.fft_size,
            "hop_size": config.hop_size,
            "sample_rate": config.sample_rate,
        }
        for name, config in _LEARNED_PRESETS.items()
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return len(data)


def load_learned_presets(path: Path | str) -> int:
    """Load learned presets from JSON file.

    Args:
        path: Input file path.

    Returns:
        Number of presets loaded.
    """
    import json

    path = Path(path)
    if not path.exists():
        return 0

    with open(path) as f:
        data = json.load(f)

    count = 0
    for name, config_data in data.items():
        _LEARNED_PRESETS[name] = AnalyzerConfig(
            fft_size=config_data.get("fft_size", 2048),
            hop_size=config_data.get("hop_size", 512),
            sample_rate=config_data.get("sample_rate", 44100),
        )
        count += 1

    return count


def get_preset_for_signal_type(
    signal_type: str, optimize_for: str = "balanced"
) -> AnalyzerConfig:
    """Get recommended preset for a specific signal type.

    Args:
        signal_type: Type of signal (tempo, onset, pitch, complex).
        optimize_for: Optimization priority (balanced, accuracy, speed, latency).

    Returns:
        Recommended AnalyzerConfig.
    """
    # Check for learned preset first
    learned_key = f"{signal_type}_{optimize_for}"
    if learned_key in _LEARNED_PRESETS:
        return _LEARNED_PRESETS[learned_key]

    # Fall back to static presets based on optimization priority
    preset_map = {
        "balanced": "real_time",
        "accuracy": "high_accuracy",
        "speed": "fast",
        "latency": "low_latency",
        "research": "research",
    }

    preset_name = preset_map.get(optimize_for, "real_time")
    return PRESETS[preset_name]

