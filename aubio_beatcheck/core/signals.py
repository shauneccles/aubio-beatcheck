"""Signal Generation Wrapper.

This module provides a unified interface to thebeat-based signal generation,
re-exporting the thebeat_generator functions with a clean API for the application.
"""

from .ground_truth import SignalDefinition
from .thebeat_gen import (
    generate_chromatic_scale,
    generate_click_track,
    generate_complex_signal,
    generate_onset_signal,
    generate_pitch_sequence,
    generate_rhythmic_pattern,
)

__all__ = [
    "SignalGenerator",
    "generate_click_track",
    "generate_onset_signal",
    "generate_pitch_sequence",
    "generate_chromatic_scale",
    "generate_rhythmic_pattern",
    "generate_complex_signal",
]


class SignalGenerator:
    """Unified interface for signal generation using thebeat.

    This class provides convenient methods for generating test signals
    with known ground truth for validating aubio analysis.
    """

    @staticmethod
    def click_track(
        bpm: float = 120.0,
        duration: float = 10.0,
        click_duration_ms: float = 50.0,
        add_timing_jitter: bool = False,
        jitter_std_ms: float = 10.0,
        rng_seed: int | None = None,
    ) -> tuple[any, SignalDefinition]:
        """Generate isochronous click track.

        Args:
            bpm: Tempo in beats per minute
            duration: Duration in seconds
            click_duration_ms: Duration of each click in milliseconds
            add_timing_jitter: Whether to add realistic timing variations
            jitter_std_ms: Standard deviation of jitter in milliseconds
            rng_seed: Random seed for reproducible jitter

        Returns:
            Tuple of (audio_array, signal_definition)
        """
        return generate_click_track(
            bpm=bpm,
            duration=duration,
            click_duration_ms=click_duration_ms,
            add_timing_jitter=add_timing_jitter,
            jitter_std_ms=jitter_std_ms,
            rng_seed=rng_seed,
        )

    @staticmethod
    def onset_signal(
        attack_type: str = "sharp",
        interval_ms: float = 500.0,
        duration: float = 10.0,
        frequency: float = 1000.0,
        waveform: str = "sine",
    ) -> tuple[any, SignalDefinition]:
        """Generate signal with various onset attack types.

        Args:
            attack_type: 'impulse', 'sharp', 'medium', or 'slow'
            interval_ms: Time between onsets in milliseconds
            duration: Duration in seconds
            frequency: Fundamental frequency in Hz
            waveform: 'sine', 'square', 'sawtooth', or 'triangle'

        Returns:
            Tuple of (audio_array, signal_definition)
        """
        return generate_onset_signal(
            attack_type=attack_type,
            interval_ms=interval_ms,
            duration=duration,
            frequency=frequency,
            waveform=waveform,
        )

    @staticmethod
    def pitch_sequence(
        midi_notes: list[int],
        note_duration: float = 0.5,
        waveform: str = "sine",
    ) -> tuple[any, SignalDefinition]:
        """Generate sequence of musical notes.

        Args:
            midi_notes: List of MIDI note numbers (21-108)
            note_duration: Duration of each note in seconds
            waveform: 'sine', 'square', 'sawtooth', or 'triangle'

        Returns:
            Tuple of (audio_array, signal_definition)
        """
        return generate_pitch_sequence(
            midi_notes=midi_notes,
            note_duration=note_duration,
            waveform=waveform,
        )

    @staticmethod
    def chromatic_scale(
        start_midi: int = 60,
        num_notes: int = 13,
        note_duration: float = 0.5,
        waveform: str = "sine",
    ) -> tuple[any, SignalDefinition]:
        """Generate chromatic scale.

        Args:
            start_midi: Starting MIDI note number
            num_notes: Number of notes in scale
            note_duration: Duration of each note in seconds
            waveform: 'sine', 'square', 'sawtooth', or 'triangle'

        Returns:
            Tuple of (audio_array, signal_definition)
        """
        return generate_chromatic_scale(
            start_midi=start_midi,
            num_notes=num_notes,
            note_duration=note_duration,
            waveform=waveform,
        )

    @staticmethod
    def rhythmic_pattern(
        pattern: str,
        grid_ioi_ms: float = 125.0,
        click_duration_ms: float = 50.0,
        frequency: float = 1000.0,
    ) -> tuple[any, SignalDefinition]:
        """Generate complex rhythmic pattern from binary notation.

        Args:
            pattern: Binary pattern string (e.g., "1001101010010110")
            grid_ioi_ms: Grid inter-onset interval in milliseconds
            click_duration_ms: Duration of each click in milliseconds
            frequency: Fundamental frequency in Hz

        Returns:
            Tuple of (audio_array, signal_definition)
        """
        return generate_rhythmic_pattern(
            pattern=pattern,
            grid_ioi_ms=grid_ioi_ms,
            click_duration_ms=click_duration_ms,
            frequency=frequency,
        )

    @staticmethod
    def complex_signal(
        bpm: float = 120.0,
        duration: float = 10.0,
        melody_notes: list[int] | None = None,
        snr_db: float = 20.0,
        rng_seed: int | None = None,
    ) -> tuple[any, SignalDefinition]:
        """Generate complex signal with beats, melody, and noise.

        Args:
            bpm: Tempo in beats per minute
            duration: Duration in seconds
            melody_notes: MIDI notes for melody (None for default)
            snr_db: Signal-to-noise ratio in decibels
            rng_seed: Random seed for reproducible noise

        Returns:
            Tuple of (audio_array, signal_definition)
        """
        return generate_complex_signal(
            bpm=bpm,
            duration=duration,
            melody_notes=melody_notes,
            snr_db=snr_db,
            rng_seed=rng_seed,
        )
