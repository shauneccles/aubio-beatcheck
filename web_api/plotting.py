"""Waveform Plotting Utilities.

Generate visualizations of audio waveforms with ground truth and detected events.
"""

import io

import matplotlib.pyplot as plt
import numpy as np


def generate_analysis_plot(
    signal_name: str,
    audio: np.ndarray,
    sample_rate: int,
    ground_truth_events: list[float],
    detected_events: list[float],
    event_type: str = "Beats",
) -> bytes:
    """Generate a plot of the waveform with ground truth and detected events.

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


def generate_pitch_analysis_plot(
    signal_name: str,
    audio: np.ndarray,
    sample_rate: int,
    ground_truth_pitches: list[dict],
    detected_pitches: list[tuple[float, float, float]],
    pitch_tolerance_cents: float = 50.0,
) -> bytes:
    """Generate a piano roll-style plot comparing ground truth and detected pitches.

    Args:
        signal_name: Name of the signal
        audio: Audio samples
        sample_rate: Sample rate in Hz
        ground_truth_pitches: List of pitch annotations with keys:
            - start_time: float
            - end_time: float
            - midi_note: int
            - frequency_hz: float (optional)
        detected_pitches: List of (time, midi_note, confidence) tuples
        pitch_tolerance_cents: Tolerance in cents for accuracy coloring

    Returns:
        Bytes of the generated PNG image
    """
    # Create figure with two subplots: waveform on top, piano roll below
    fig, (ax_wave, ax_pitch) = plt.subplots(
        2, 1, figsize=(12, 6), height_ratios=[1, 2], sharex=True
    )

    # Create time axis
    duration = len(audio) / sample_rate
    time_axis = np.linspace(0, duration, len(audio))

    # --- Top subplot: Waveform ---
    ax_wave.plot(time_axis, audio, color="steelblue", alpha=0.7, linewidth=0.5)
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title(f"Pitch Analysis: {signal_name}")
    ax_wave.grid(True, alpha=0.3)
    ax_wave.set_xlim(0, duration)

    # --- Bottom subplot: Piano Roll ---

    # Collect all MIDI notes for y-axis range
    all_midi_notes = []

    # Plot ground truth as horizontal bars
    for gt in ground_truth_pitches:
        start = gt.get("start_time", 0)
        end = gt.get("end_time", start + 0.1)
        midi = gt.get("midi_note", 60)
        all_midi_notes.append(midi)

        # Draw a horizontal bar for the expected pitch
        ax_pitch.barh(
            y=midi,
            width=end - start,
            left=start,
            height=0.6,
            color="green",
            alpha=0.4,
            edgecolor="darkgreen",
            linewidth=1,
            label="Ground Truth" if gt == ground_truth_pitches[0] else None,
        )

    # Convert tolerance from cents to MIDI note units
    # 100 cents = 1 semitone = 1 MIDI note
    tolerance_midi = pitch_tolerance_cents / 100.0

    # Plot detected pitches as scatter points
    if detected_pitches:
        det_times = [p[0] for p in detected_pitches]
        det_notes = [p[1] for p in detected_pitches]
        det_conf = [p[2] for p in detected_pitches]
        all_midi_notes.extend(det_notes)

        # Determine accuracy: check if each detection is within tolerance of ground truth
        colors = []
        for t, note in zip(det_times, det_notes, strict=False):
            # Find ground truth pitch at this time
            expected_note = None
            for gt in ground_truth_pitches:
                if gt["start_time"] <= t < gt["end_time"]:
                    expected_note = gt["midi_note"]
                    break

            if expected_note is not None:
                error = abs(note - expected_note)
                if error <= tolerance_midi:
                    colors.append("limegreen")  # Accurate
                else:
                    colors.append("red")  # Inaccurate
            else:
                colors.append("orange")  # No ground truth at this time

        # Size points by confidence
        sizes = [max(10, min(50, c * 50)) for c in det_conf]

        ax_pitch.scatter(
            det_times,
            det_notes,
            c=colors,
            s=sizes,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

    # Configure y-axis
    if all_midi_notes:
        min_note = max(0, int(min(all_midi_notes)) - 2)
        max_note = min(127, int(max(all_midi_notes)) + 2)
    else:
        min_note, max_note = 55, 75  # Default range around middle C

    ax_pitch.set_ylim(min_note, max_note)
    ax_pitch.set_xlim(0, duration)

    # Add MIDI note labels with note names
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    yticks = list(range(min_note, max_note + 1))

    # Only show labels for every note if range is small, else show fewer
    if max_note - min_note <= 24:
        # Show C notes with octave numbers
        yticklabels = []
        for n in yticks:
            name = note_names[n % 12]
            octave = (n // 12) - 1
            if name == "C":
                yticklabels.append(f"C{octave}")
            else:
                yticklabels.append(f"{n}")
        ax_pitch.set_yticks(yticks)
        ax_pitch.set_yticklabels(yticklabels, fontsize=8)
    else:
        # Show only C notes
        c_notes = [n for n in yticks if n % 12 == 0]
        ax_pitch.set_yticks(c_notes)
        ax_pitch.set_yticklabels([f"C{(n // 12) - 1}" for n in c_notes])

    ax_pitch.set_xlabel("Time (s)")
    ax_pitch.set_ylabel("MIDI Note")
    ax_pitch.grid(True, alpha=0.3, axis="both")

    # Add legend with custom handles
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="green", alpha=0.4, edgecolor="darkgreen", label="Ground Truth"
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="limegreen",
            markersize=8,
            markeredgecolor="black",
            label="Accurate Detection",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=8,
            markeredgecolor="black",
            label="Inaccurate Detection",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="orange",
            markersize=8,
            markeredgecolor="black",
            label="No Ground Truth",
        ),
    ]
    ax_pitch.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close()

    buf.seek(0)
    return buf.getvalue()
