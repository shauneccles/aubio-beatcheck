"""
Standard Test Suites for Aubio Validation

This module defines comprehensive test suites for validating aubio's
tempo, onset, pitch, and pvoc functionality using thebeat-generated signals.
"""

from dataclasses import dataclass
import numpy as np

from aubio_beatcheck.core.ground_truth import SignalDefinition
from aubio_beatcheck.core.thebeat_gen import (
    generate_chromatic_scale,
    generate_click_track,
    generate_complex_signal,
    generate_onset_signal,
    generate_pitch_sequence,
    generate_rhythmic_pattern,
)


@dataclass
class TestSignal:
    """A single test signal with audio and ground truth."""

    name: str
    description: str
    audio: np.ndarray
    signal_def: SignalDefinition
    category: str  # 'tempo', 'onset', 'pitch', 'complex'


class StandardSuites:
    """
    Pre-defined test suites for comprehensive aubio validation.

    Each suite contains multiple signals designed to test specific
    aspects of aubio's analysis capabilities.
    """

    @staticmethod
    def tempo_suite(duration: float = 10.0) -> list[TestSignal]:
        """
        Tempo/beat tracking test suite.

        Tests tempo detection across BPM range with various conditions:
        - Slow (60 BPM), moderate (120 BPM), fast (180 BPM)
        - Timing jitter (realistic human variation)
        - Different click durations

        Args:
            duration: Duration of each signal in seconds

        Returns:
            List of TestSignal objects
        """
        signals = []

        # Basic tempo range tests
        for bpm in [60, 90, 120, 140, 160, 180]:
            audio, signal_def = generate_click_track(bpm=bpm, duration=duration)
            signals.append(
                TestSignal(
                    name=f"click_{bpm}bpm",
                    description=f"{bpm} BPM isochronous click track",
                    audio=audio,
                    signal_def=signal_def,
                    category="tempo",
                )
            )

        # Timing jitter tests (human-like variation)
        for jitter_ms in [5, 10, 15]:
            audio, signal_def = generate_click_track(
                bpm=120,
                duration=duration,
                add_timing_jitter=True,
                jitter_std_ms=jitter_ms,
                rng_seed=42,
            )
            signals.append(
                TestSignal(
                    name=f"click_120bpm_jitter{jitter_ms}ms",
                    description=f"120 BPM with ±{jitter_ms}ms timing jitter",
                    audio=audio,
                    signal_def=signal_def,
                    category="tempo",
                )
            )

        # Variable click durations
        for click_dur_ms in [20, 50, 100]:
            audio, signal_def = generate_click_track(
                bpm=120, duration=duration, click_duration_ms=click_dur_ms
            )
            signals.append(
                TestSignal(
                    name=f"click_120bpm_{click_dur_ms}ms_duration",
                    description=f"120 BPM with {click_dur_ms}ms click duration",
                    audio=audio,
                    signal_def=signal_def,
                    category="tempo",
                )
            )

        return signals

    @staticmethod
    def onset_suite(duration: float = 10.0) -> list[TestSignal]:
        """
        Onset detection test suite.

        Tests onset detection with various attack characteristics:
        - Attack types: impulse, sharp, medium, slow
        - Different onset intervals
        - Different waveforms

        Args:
            duration: Duration of each signal in seconds

        Returns:
            List of TestSignal objects
        """
        signals = []

        # Attack type tests
        for attack_type in ["impulse", "sharp", "medium", "slow"]:
            audio, signal_def = generate_onset_signal(
                attack_type=attack_type, interval_ms=500.0, duration=duration
            )
            signals.append(
                TestSignal(
                    name=f"onset_{attack_type}_500ms",
                    description=f"{attack_type.capitalize()} onset every 500ms",
                    audio=audio,
                    signal_def=signal_def,
                    category="onset",
                )
            )

        # Interval variation tests (sharp attacks)
        for interval_ms in [250, 500, 1000]:
            audio, signal_def = generate_onset_signal(
                attack_type="sharp", interval_ms=interval_ms, duration=duration
            )
            signals.append(
                TestSignal(
                    name=f"onset_sharp_{interval_ms}ms",
                    description=f"Sharp onset every {interval_ms}ms",
                    audio=audio,
                    signal_def=signal_def,
                    category="onset",
                )
            )

        # Waveform tests (sharp attacks)
        for waveform in ["sine", "square", "sawtooth", "triangle"]:
            audio, signal_def = generate_onset_signal(
                attack_type="sharp",
                interval_ms=500.0,
                duration=duration,
                waveform=waveform,
            )
            signals.append(
                TestSignal(
                    name=f"onset_sharp_{waveform}",
                    description=f"Sharp {waveform} onset every 500ms",
                    audio=audio,
                    signal_def=signal_def,
                    category="onset",
                )
            )

        return signals

    @staticmethod
    def pitch_suite() -> list[TestSignal]:
        """
        Pitch detection test suite.

        Tests pitch detection across musical range:
        - Chromatic scales (different octaves)
        - Musical intervals
        - Different waveforms

        Returns:
            List of TestSignal objects
        """
        signals = []

        # Chromatic scales at different octaves
        for start_midi, octave_name in [(36, "low"), (60, "middle"), (84, "high")]:
            audio, signal_def = generate_chromatic_scale(
                start_midi=start_midi, num_notes=13, note_duration=0.5
            )
            signals.append(
                TestSignal(
                    name=f"chromatic_{octave_name}_octave",
                    description=f"Chromatic scale starting at MIDI {start_midi}",
                    audio=audio,
                    signal_def=signal_def,
                    category="pitch",
                )
            )

        # Musical intervals (C major scale)
        c_major = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5
        audio, signal_def = generate_pitch_sequence(
            midi_notes=c_major, note_duration=0.5
        )
        signals.append(
            TestSignal(
                name="c_major_scale",
                description="C major scale (C4 to C5)",
                audio=audio,
                signal_def=signal_def,
                category="pitch",
            )
        )

        # Chord progression (C major, F major, G major, C major)
        chords = [
            [60, 64, 67],  # C major
            [65, 69, 72],  # F major
            [67, 71, 74],  # G major
            [60, 64, 67],  # C major
        ]
        for i, chord in enumerate(chords):
            for note in chord:
                audio, signal_def = generate_pitch_sequence(
                    midi_notes=[note] * 4, note_duration=0.25
                )
                # Note: This is simplified - proper chord test would mix signals
                signals.append(
                    TestSignal(
                        name=f"chord_{i}_note_{note}",
                        description=f"Chord {i+1}, MIDI note {note}",
                        audio=audio,
                        signal_def=signal_def,
                        category="pitch",
                    )
                )

        # Waveform tests (same pitch, different timbres)
        for waveform in ["sine", "square", "sawtooth", "triangle"]:
            audio, signal_def = generate_pitch_sequence(
                midi_notes=[60] * 10, note_duration=0.5, waveform=waveform
            )
            signals.append(
                TestSignal(
                    name=f"pitch_{waveform}_c4",
                    description=f"C4 ({waveform} wave) repeated",
                    audio=audio,
                    signal_def=signal_def,
                    category="pitch",
                )
            )

        return signals

    @staticmethod
    def rhythmic_pattern_suite() -> list[TestSignal]:
        """
        Rhythmic pattern test suite.

        Tests complex rhythmic patterns:
        - Syncopation
        - Polyrhythms
        - Various pattern densities

        Returns:
            List of TestSignal objects
        """
        signals = []

        patterns = [
            ("simple_4_4", "1000100010001000", "Simple 4/4 kick pattern"),
            ("syncopated", "1001101010010110", "Syncopated 16-step pattern"),
            ("sparse", "1000000010000000", "Sparse pattern (2 hits per bar)"),
            ("dense", "1111111111111111", "Dense pattern (all 16ths)"),
            ("clave_son", "1001000100100000", "Son clave pattern"),
        ]

        for name, pattern, desc in patterns:
            audio, signal_def = generate_rhythmic_pattern(
                pattern=pattern, grid_ioi_ms=125.0  # 16th notes at 120 BPM
            )
            signals.append(
                TestSignal(
                    name=f"rhythm_{name}",
                    description=desc,
                    audio=audio,
                    signal_def=signal_def,
                    category="tempo",
                )
            )

        return signals

    @staticmethod
    def complex_suite(duration: float = 10.0) -> list[TestSignal]:
        """
        Complex signal test suite.

        Tests combined elements with realistic noise:
        - Beats + melody + noise at various SNR levels
        - Tests robustness to interference

        Args:
            duration: Duration of each signal in seconds

        Returns:
            List of TestSignal objects
        """
        signals = []

        # Various SNR levels
        for snr_db in [10, 20, 30]:
            audio, signal_def = generate_complex_signal(
                bpm=120, duration=duration, snr_db=snr_db, rng_seed=42
            )
            signals.append(
                TestSignal(
                    name=f"complex_120bpm_snr{snr_db}db",
                    description=f"Complex signal (beats+melody+noise) at {snr_db}dB SNR",
                    audio=audio,
                    signal_def=signal_def,
                    category="complex",
                )
            )

        # Different tempos with moderate SNR
        for bpm in [90, 140, 160]:
            audio, signal_def = generate_complex_signal(
                bpm=bpm, duration=duration, snr_db=20, rng_seed=42
            )
            signals.append(
                TestSignal(
                    name=f"complex_{bpm}bpm_snr20db",
                    description=f"Complex signal at {bpm} BPM with 20dB SNR",
                    audio=audio,
                    signal_def=signal_def,
                    category="complex",
                )
            )

        return signals

    @classmethod
    def all_suites(cls, duration: float = 10.0) -> dict[str, list[TestSignal]]:
        """
        Get all standard test suites.

        Args:
            duration: Duration for tempo/onset/complex suites

        Returns:
            Dictionary mapping suite names to signal lists
        """
        return {
            "tempo": cls.tempo_suite(duration),
            "onset": cls.onset_suite(duration),
            "pitch": cls.pitch_suite(),
            "rhythmic": cls.rhythmic_pattern_suite(),
            "complex": cls.complex_suite(duration),
        }

    @classmethod
    def get_suite(cls, suite_name: str, duration: float = 10.0) -> list[TestSignal]:
        """
        Get a specific test suite by name.

        Args:
            suite_name: Name of suite ('tempo', 'onset', 'pitch', 'rhythmic', 'complex', 'all')
            duration: Duration for applicable suites

        Returns:
            List of TestSignal objects

        Raises:
            ValueError: If suite_name is not recognized
        """
        suite_map = {
            "tempo": cls.tempo_suite,
            "onset": cls.onset_suite,
            "pitch": cls.pitch_suite,
            "rhythmic": cls.rhythmic_pattern_suite,
            "complex": cls.complex_suite,
        }

        if suite_name == "all":
            all_signals = []
            for suite_func in suite_map.values():
                # Check if function accepts duration parameter
                import inspect

                sig = inspect.signature(suite_func)
                if "duration" in sig.parameters:
                    all_signals.extend(suite_func(duration))
                else:
                    all_signals.extend(suite_func())
            return all_signals

        if suite_name not in suite_map:
            raise ValueError(
                f"Unknown suite: {suite_name}. "
                f"Available: {', '.join(suite_map.keys())}, 'all'"
            )

        suite_func = suite_map[suite_name]
        import inspect

        sig = inspect.signature(suite_func)
        if "duration" in sig.parameters:
            return suite_func(duration)
        else:
            return suite_func()
