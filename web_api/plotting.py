import io
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def generate_analysis_plot(
    signal_name: str,
    audio: np.ndarray,
    sample_rate: int,
    ground_truth_events: List[float],
    detected_events: List[float],
    event_type: str = "Beats",
) -> bytes:
    """
    Generate a plot of the waveform with ground truth and detected events.

    Args:
        signal_name: Name of the signal
        audio: Audio samples
        sample_rate: Sample rate in Hz
        ground_truth_events: List of timestamps for ground truth events
        detected_events: List of timestamps for detected events
        event_type: Label for events (e.g., "Beats", "Onsets")

    Returns:
        Bytes of the generated PNG image
    """
    # Create figure
    plt.figure(figsize=(10, 4))

    # Create time axis
    duration = len(audio) / sample_rate
    time_axis = np.linspace(0, duration, len(audio))

    # Plot waveform
    plt.plot(time_axis, audio, color="lightgray", label="Waveform", alpha=0.7)

    # Plot ground truth
    for i, t in enumerate(ground_truth_events):
        label = "Ground Truth" if i == 0 else None
        plt.axvline(x=t, color="green", linestyle="-", alpha=0.8, label=label)

    # Plot detected
    for i, t in enumerate(detected_events):
        label = "Detected" if i == 0 else None
        plt.axvline(x=t, color="red", linestyle="--", alpha=0.8, label=label)

    plt.title(f"Analysis: {signal_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close()

    buf.seek(0)
    return buf.getvalue()
