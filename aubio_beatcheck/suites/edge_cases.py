"""Edge Case Test Suites for Aubio Validation.

Comprehensive edge case and stress test signals designed to expose
analyzer weaknesses and validate robustness under challenging conditions.
"""

import numpy as np

from aubio_beatcheck.core.ground_truth import (
    BeatAnnotation,
    GroundTruth,
    OnsetAnnotation,
    SignalDefinition,
    SignalMetadata,
    TestCriteria,
)
from aubio_beatcheck.core.thebeat_gen import (
    DEFAULT_SAMPLE_RATE,
    generate_click_track,
    generate_onset_signal,
)
from aubio_beatcheck.suites.standard import TestSignal


class EdgeCaseSuites:
    """Edge case test suites for exposing analyzer weaknesses.

    These signals are designed to test boundary conditions, stress
    scenarios, and real-world imperfections that standard test
    suites don't cover.
    """

    @staticmethod
    def tempo_edge_cases(duration: float = 10.0) -> list[TestSignal]:
        """Tempo detection edge cases.

        Tests tempo detection at boundary conditions:
        - Very slow tempo (30 BPM) - below typical detection range
        - Very fast tempo (240 BPM) - near upper detection limit
        - Tempo ramp - gradual tempo change
        - Sudden tempo change - abrupt BPM switch
        - Polyrhythm - competing beat patterns
        - Offbeat emphasis - weak beats emphasized

        Args:
            duration: Duration of each signal in seconds

        Returns:
            List of TestSignal objects
        """
        signals = []

        # Very slow tempo - 30 BPM (2 second intervals)
        audio, signal_def = generate_click_track(
            bpm=30,
            duration=max(duration, 20.0),  # Need longer for slow tempo
        )
        signals.append(
            TestSignal(
                name="tempo_very_slow_30bpm",
                description="Very slow 30 BPM - tests low BPM detection boundary",
                audio=audio,
                signal_def=signal_def,
                category="tempo",
            )
        )

        # Very fast tempo - 240 BPM
        audio, signal_def = generate_click_track(bpm=240, duration=duration)
        signals.append(
            TestSignal(
                name="tempo_very_fast_240bpm",
                description="Very fast 240 BPM - tests high BPM detection limit",
                audio=audio,
                signal_def=signal_def,
                category="tempo",
            )
        )

        # Tempo ramp: 80 BPM to 160 BPM over duration
        audio, signal_def = _generate_tempo_ramp(
            start_bpm=80, end_bpm=160, duration=duration
        )
        signals.append(
            TestSignal(
                name="tempo_ramp_80_to_160",
                description="Gradual tempo increase from 80 to 160 BPM",
                audio=audio,
                signal_def=signal_def,
                category="tempo",
            )
        )

        # Sudden tempo change at midpoint
        audio, signal_def = _generate_tempo_change(
            bpm1=100, bpm2=150, change_time=duration / 2, duration=duration
        )
        signals.append(
            TestSignal(
                name="tempo_sudden_change",
                description="Sudden tempo change from 100 to 150 BPM at midpoint",
                audio=audio,
                signal_def=signal_def,
                category="tempo",
            )
        )

        # Polyrhythm: 3 against 4 pattern
        audio, signal_def = _generate_polyrhythm(
            primary_bpm=120, secondary_bpm=90, duration=duration
        )
        signals.append(
            TestSignal(
                name="tempo_polyrhythm_3_4",
                description="Polyrhythm with competing 120 and 90 BPM patterns",
                audio=audio,
                signal_def=signal_def,
                category="tempo",
            )
        )

        # Intermittent beats with silence gaps
        audio, signal_def = _generate_intermittent_beats(
            bpm=120, gap_duration=2.0, duration=duration
        )
        signals.append(
            TestSignal(
                name="tempo_intermittent_gaps",
                description="120 BPM with 2-second silence gaps",
                audio=audio,
                signal_def=signal_def,
                category="tempo",
            )
        )

        # Extreme timing jitter
        audio, signal_def = generate_click_track(
            bpm=120,
            duration=duration,
            add_timing_jitter=True,
            jitter_std_ms=30.0,  # Very high jitter
            rng_seed=42,
        )
        signals.append(
            TestSignal(
                name="tempo_extreme_jitter_30ms",
                description="120 BPM with extreme ±30ms timing jitter",
                audio=audio,
                signal_def=signal_def,
                category="tempo",
            )
        )

        # Very short signal - boundary condition
        audio, signal_def = generate_click_track(bpm=120, duration=2.0)
        signals.append(
            TestSignal(
                name="tempo_very_short_2sec",
                description="Very short 2-second signal at 120 BPM",
                audio=audio,
                signal_def=signal_def,
                category="tempo",
            )
        )

        return signals

    @staticmethod
    def onset_edge_cases(duration: float = 10.0) -> list[TestSignal]:
        """Onset detection edge cases.

        Tests onset detection with challenging characteristics:
        - Dense onsets - very rapid successive onsets
        - Ultra-slow attack - gradual onset that's hard to pinpoint
        - Double strikes - flam-like double hits
        - Mixed attack types - varying attacks in same signal
        - Frequency sweep onset - spectral change as onset cue

        Args:
            duration: Duration of each signal in seconds

        Returns:
            List of TestSignal objects
        """
        signals = []

        # Very dense onsets - 20 per second (50ms intervals) - use custom generator
        audio, signal_def = _generate_dense_onsets(interval_ms=50.0, duration=duration)
        signal_def.test_criteria.onset_timing_tolerance_ms = 20.0  # Tighter tolerance
        signals.append(
            TestSignal(
                name="onset_very_dense_50ms",
                description="Very dense onsets at 50ms intervals (20/sec)",
                audio=audio,
                signal_def=signal_def,
                category="onset",
            )
        )

        # Ultra-slow attack - 200ms attack time
        audio, signal_def = _generate_ultra_slow_onset(
            attack_ms=200.0, interval_ms=500.0, duration=duration
        )
        signals.append(
            TestSignal(
                name="onset_ultra_slow_200ms",
                description="Ultra-slow 200ms attack time - hard to localize",
                audio=audio,
                signal_def=signal_def,
                category="onset",
            )
        )

        # Double strikes (flam-like)
        audio, signal_def = _generate_double_onset(
            interval_ms=500.0, secondary_delay_ms=20.0, duration=duration
        )
        signals.append(
            TestSignal(
                name="onset_double_strike_flam",
                description="Double strikes with 20ms delay (flam)",
                audio=audio,
                signal_def=signal_def,
                category="onset",
            )
        )

        # Mixed attack types in same signal
        audio, signal_def = _generate_mixed_attacks(duration=duration)
        signals.append(
            TestSignal(
                name="onset_mixed_attacks",
                description="Mixed sharp, medium, and slow attacks",
                audio=audio,
                signal_def=signal_def,
                category="onset",
            )
        )

        # Sparse onsets - 2 second intervals
        audio, signal_def = generate_onset_signal(
            attack_type="sharp", interval_ms=2000.0, duration=duration
        )
        signals.append(
            TestSignal(
                name="onset_very_sparse_2sec",
                description="Very sparse onsets at 2-second intervals",
                audio=audio,
                signal_def=signal_def,
                category="onset",
            )
        )

        # Onsets with frequency sweep
        audio, signal_def = _generate_sweep_onset(
            start_freq=200.0, end_freq=2000.0, interval_ms=500.0, duration=duration
        )
        signals.append(
            TestSignal(
                name="onset_frequency_sweep",
                description="Onsets with frequency sweep from 200-2000 Hz",
                audio=audio,
                signal_def=signal_def,
                category="onset",
            )
        )

        return signals

    @staticmethod
    def robustness_suite(duration: float = 10.0) -> list[TestSignal]:
        """Robustness stress tests for real-world conditions.

        Tests analyzer robustness against:
        - Extreme noise (0 dB SNR)
        - High noise (5 dB SNR)
        - Clipped audio
        - DC offset
        - Very low amplitude
        - Phase inverted signal

        Args:
            duration: Duration of each signal in seconds

        Returns:
            List of TestSignal objects
        """
        signals = []

        # Extreme noise - 0 dB SNR (signal equals noise)
        audio, signal_def = _generate_noisy_signal(
            bpm=120, snr_db=0.0, duration=duration
        )
        signals.append(
            TestSignal(
                name="robustness_extreme_noise_0db",
                description="Extreme noise at 0 dB SNR (equal signal and noise)",
                audio=audio,
                signal_def=signal_def,
                category="tempo",
            )
        )

        # High noise - 5 dB SNR
        audio, signal_def = _generate_noisy_signal(
            bpm=120, snr_db=5.0, duration=duration
        )
        signals.append(
            TestSignal(
                name="robustness_high_noise_5db",
                description="High noise at 5 dB SNR",
                audio=audio,
                signal_def=signal_def,
                category="tempo",
            )
        )

        # Clipped audio - 10% of peaks clipped
        audio, signal_def = _generate_clipped_signal(
            bpm=120, clip_threshold=0.7, duration=duration
        )
        signals.append(
            TestSignal(
                name="robustness_clipped_audio",
                description="Clipped audio (peaks at 70% threshold)",
                audio=audio,
                signal_def=signal_def,
                category="tempo",
            )
        )

        # DC offset
        audio, signal_def = _generate_dc_offset_signal(
            bpm=120, dc_offset=0.3, duration=duration
        )
        signals.append(
            TestSignal(
                name="robustness_dc_offset",
                description="Signal with 0.3 DC offset",
                audio=audio,
                signal_def=signal_def,
                category="tempo",
            )
        )

        # Very low amplitude
        audio, signal_def = _generate_low_amplitude_signal(
            bpm=120, amplitude=0.05, duration=duration
        )
        signals.append(
            TestSignal(
                name="robustness_low_amplitude",
                description="Very low amplitude signal (5% of full scale)",
                audio=audio,
                signal_def=signal_def,
                category="tempo",
            )
        )

        # Phase inverted
        audio, signal_def = generate_click_track(bpm=120, duration=duration)
        audio = -audio  # Invert phase
        signals.append(
            TestSignal(
                name="robustness_phase_inverted",
                description="Phase-inverted signal",
                audio=audio,
                signal_def=signal_def,
                category="tempo",
            )
        )

        # Quantization noise (low bit depth simulation)
        audio, signal_def = _generate_quantized_signal(
            bpm=120, bit_depth=8, duration=duration
        )
        signals.append(
            TestSignal(
                name="robustness_low_bitdepth_8bit",
                description="Low bit depth (8-bit) quantization",
                audio=audio,
                signal_def=signal_def,
                category="tempo",
            )
        )

        # Bandlimited signal (simulating phone/low quality audio)
        audio, signal_def = _generate_bandlimited_signal(
            bpm=120, low_freq=300.0, high_freq=3400.0, duration=duration
        )
        signals.append(
            TestSignal(
                name="robustness_bandlimited_phone",
                description="Phone-quality bandlimited (300-3400 Hz)",
                audio=audio,
                signal_def=signal_def,
                category="tempo",
            )
        )

        return signals

    @classmethod
    def all_edge_cases(cls, duration: float = 10.0) -> list[TestSignal]:
        """Get all edge case test signals.

        Args:
            duration: Duration for applicable suites

        Returns:
            List of all TestSignal objects from all edge case suites
        """
        return (
            cls.tempo_edge_cases(duration)
            + cls.onset_edge_cases(duration)
            + cls.robustness_suite(duration)
        )


# --- Helper Functions for Edge Case Signal Generation ---


def _generate_dense_onsets(
    interval_ms: float,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, SignalDefinition]:
    """Generate very dense onsets with short intervals.

    Uses a simple click generator that works with short intervals
    where thebeat's SoundSequence would fail.
    """
    total_samples = int(duration * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float32)
    onsets = []

    current_time_ms = 0.0
    # Use very short click to fit in small intervals
    click_duration_ms = min(10.0, interval_ms * 0.3)
    click_samples = int(click_duration_ms * sample_rate / 1000)

    while current_time_ms < duration * 1000:
        onsets.append(
            OnsetAnnotation(
                time=current_time_ms / 1000.0,
                attack_ms=0.5,
                onset_type="impulse",
            )
        )

        # Generate short click
        start_sample = int(current_time_ms * sample_rate / 1000)
        end_sample = min(start_sample + click_samples, total_samples)

        if end_sample > start_sample:
            t = np.arange(end_sample - start_sample) / sample_rate
            click = np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 200)
            audio[start_sample:end_sample] += click.astype(np.float32)

        current_time_ms += interval_ms

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    signal_def = SignalDefinition(
        signal_type="onset",
        metadata=SignalMetadata(
            sample_rate=sample_rate,
            duration=duration,
            description=f"Dense onsets at {interval_ms}ms intervals",
        ),
        ground_truth=GroundTruth(onsets=onsets),
        test_criteria=TestCriteria(
            onset_timing_tolerance_ms=interval_ms / 2,
            min_detection_rate=0.80,
        ),
    )

    return audio.astype(np.float32), signal_def


def _generate_tempo_ramp(
    start_bpm: float,
    end_bpm: float,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, SignalDefinition]:
    """Generate a signal with gradually changing tempo."""
    # Calculate variable IOIs for tempo ramp
    total_samples = int(duration * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float32)

    beats = []
    current_time = 0.0
    beat_num = 0

    while current_time < duration:
        # Interpolate BPM at current position
        progress = current_time / duration
        current_bpm = start_bpm + (end_bpm - start_bpm) * progress
        current_ioi = 60.0 / current_bpm

        # Add beat annotation
        beats.append(
            BeatAnnotation(
                time=current_time,
                beat_number=(beat_num % 4) + 1,
                bar=(beat_num // 4) + 1,
            )
        )

        # Generate click at this position
        click_duration = 0.01  # 10ms click
        click_samples = int(click_duration * sample_rate)
        start_sample = int(current_time * sample_rate)
        end_sample = min(start_sample + click_samples, total_samples)

        # Generate sine click with envelope
        t = np.arange(end_sample - start_sample) / sample_rate
        click = np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 50)
        audio[start_sample:end_sample] += click.astype(np.float32)

        current_time += current_ioi
        beat_num += 1

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    signal_def = SignalDefinition(
        signal_type="tempo",
        metadata=SignalMetadata(
            sample_rate=sample_rate,
            duration=duration,
            description=f"Tempo ramp from {start_bpm} to {end_bpm} BPM",
            bpm=(start_bpm + end_bpm) / 2,  # Average BPM
        ),
        ground_truth=GroundTruth(beats=beats),
        test_criteria=TestCriteria(
            tempo_tolerance_bpm=10.0,  # Higher tolerance for changing tempo
            beat_timing_tolerance_ms=75.0,
            min_detection_rate=0.80,
        ),
    )

    return audio.astype(np.float32), signal_def


def _generate_tempo_change(
    bpm1: float,
    bpm2: float,
    change_time: float,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, SignalDefinition]:
    """Generate a signal with sudden tempo change."""
    # Generate first section
    audio1, sig1 = generate_click_track(bpm=bpm1, duration=change_time)

    # Generate second section
    audio2, sig2 = generate_click_track(bpm=bpm2, duration=duration - change_time)

    # Combine
    audio = np.concatenate([audio1, audio2])

    # Combine beat annotations (shift second section times)
    beats = sig1.ground_truth.beats.copy()
    for beat in sig2.ground_truth.beats:
        shifted_beat = BeatAnnotation(
            time=beat.time + change_time,
            beat_number=beat.beat_number,
            bar=beat.bar + (len(sig1.ground_truth.beats) // 4),
        )
        beats.append(shifted_beat)

    signal_def = SignalDefinition(
        signal_type="tempo",
        metadata=SignalMetadata(
            sample_rate=sample_rate,
            duration=duration,
            description=f"Sudden tempo change: {bpm1} to {bpm2} BPM at {change_time}s",
            bpm=bpm1,  # Initial BPM
        ),
        ground_truth=GroundTruth(beats=beats),
        test_criteria=TestCriteria(
            tempo_tolerance_bpm=5.0,
            beat_timing_tolerance_ms=60.0,
            min_detection_rate=0.85,
        ),
    )

    return audio.astype(np.float32), signal_def


def _generate_polyrhythm(
    primary_bpm: float,
    secondary_bpm: float,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, SignalDefinition]:
    """Generate a polyrhythmic signal with two competing tempos."""
    # Generate both patterns
    audio1, sig1 = generate_click_track(
        bpm=primary_bpm, duration=duration, click_freq=1000.0
    )
    audio2, _ = generate_click_track(
        bpm=secondary_bpm,
        duration=duration,
        click_freq=500.0,  # Different frequency
    )

    # Handle different array lengths by padding shorter one
    max_len = max(len(audio1), len(audio2))
    if len(audio1) < max_len:
        audio1 = np.pad(audio1, (0, max_len - len(audio1)), mode="constant")
    if len(audio2) < max_len:
        audio2 = np.pad(audio2, (0, max_len - len(audio2)), mode="constant")

    # Mix with primary being louder
    audio = 0.7 * audio1 + 0.3 * audio2

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    # Use primary pattern as ground truth
    signal_def = SignalDefinition(
        signal_type="tempo",
        metadata=SignalMetadata(
            sample_rate=sample_rate,
            duration=duration,
            description=f"Polyrhythm: {primary_bpm} BPM (primary) vs {secondary_bpm} BPM",
            bpm=primary_bpm,
        ),
        ground_truth=sig1.ground_truth,
        test_criteria=TestCriteria(
            tempo_tolerance_bpm=5.0,
            beat_timing_tolerance_ms=50.0,
            min_detection_rate=0.70,  # Lower rate expected due to competing pattern
        ),
    )

    return audio.astype(np.float32), signal_def


def _generate_intermittent_beats(
    bpm: float,
    gap_duration: float,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, SignalDefinition]:
    """Generate beats with silence gaps."""
    ioi = 60.0 / bpm
    segment_duration = gap_duration  # Beats for this long, then silence

    total_samples = int(duration * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float32)
    beats = []

    is_playing = True
    current_time = 0.0
    segment_start = 0.0
    beat_num = 0

    while current_time < duration:
        if is_playing:
            # Add beat
            beats.append(
                BeatAnnotation(
                    time=current_time,
                    beat_number=(beat_num % 4) + 1,
                    bar=(beat_num // 4) + 1,
                )
            )

            # Generate click
            click_samples = int(0.01 * sample_rate)
            start_sample = int(current_time * sample_rate)
            end_sample = min(start_sample + click_samples, total_samples)

            t = np.arange(end_sample - start_sample) / sample_rate
            click = np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 50)
            audio[start_sample:end_sample] += click.astype(np.float32)

            beat_num += 1
            current_time += ioi

            # Check if segment is over
            if current_time - segment_start >= segment_duration:
                is_playing = False
                segment_start = current_time
        else:
            # Silence gap
            current_time += gap_duration
            is_playing = True
            segment_start = current_time

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    signal_def = SignalDefinition(
        signal_type="tempo",
        metadata=SignalMetadata(
            sample_rate=sample_rate,
            duration=duration,
            description=f"{bpm} BPM with {gap_duration}s silence gaps",
            bpm=bpm,
        ),
        ground_truth=GroundTruth(beats=beats),
        test_criteria=TestCriteria(
            tempo_tolerance_bpm=3.0,
            beat_timing_tolerance_ms=50.0,
            min_detection_rate=0.80,
        ),
    )

    return audio.astype(np.float32), signal_def


def _generate_ultra_slow_onset(
    attack_ms: float,
    interval_ms: float,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, SignalDefinition]:
    """Generate onsets with very slow attack time."""
    total_samples = int(duration * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float32)
    onsets = []

    current_time_ms = 0.0
    while current_time_ms < duration * 1000:
        onsets.append(
            OnsetAnnotation(
                time=current_time_ms / 1000.0,
                attack_ms=attack_ms,
                onset_type="slow",
            )
        )

        # Generate tone with slow attack
        onset_duration_ms = attack_ms + 200.0
        onset_samples = int(onset_duration_ms * sample_rate / 1000)
        start_sample = int(current_time_ms * sample_rate / 1000)
        end_sample = min(start_sample + onset_samples, total_samples)

        t = np.arange(end_sample - start_sample) / sample_rate

        # Slow attack envelope
        attack_time = attack_ms / 1000.0
        envelope = np.minimum(t / attack_time, 1.0)
        decay_start = attack_time + 0.1
        envelope *= np.exp(-np.maximum(t - decay_start, 0) * 10)

        tone = np.sin(2 * np.pi * 1000 * t) * envelope
        audio[start_sample:end_sample] += tone.astype(np.float32)

        current_time_ms += interval_ms

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    signal_def = SignalDefinition(
        signal_type="onset",
        metadata=SignalMetadata(
            sample_rate=sample_rate,
            duration=duration,
            description=f"Ultra-slow {attack_ms}ms attack at {interval_ms}ms intervals",
        ),
        ground_truth=GroundTruth(onsets=onsets),
        test_criteria=TestCriteria(
            onset_timing_tolerance_ms=attack_ms,  # Tolerance matches attack time
            min_detection_rate=0.70,  # Lower expected rate
        ),
    )

    return audio.astype(np.float32), signal_def


def _generate_double_onset(
    interval_ms: float,
    secondary_delay_ms: float,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, SignalDefinition]:
    """Generate double-strike (flam) onsets."""
    total_samples = int(duration * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float32)
    onsets = []

    current_time_ms = 0.0
    click_samples = int(0.01 * sample_rate)

    while current_time_ms < duration * 1000:
        # Primary onset (ground truth)
        onsets.append(
            OnsetAnnotation(
                time=current_time_ms / 1000.0,
                attack_ms=1.0,
                onset_type="sharp",
            )
        )

        # Generate primary click
        start_sample = int(current_time_ms * sample_rate / 1000)
        end_sample = min(start_sample + click_samples, total_samples)
        t = np.arange(end_sample - start_sample) / sample_rate
        click = np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 100)
        audio[start_sample:end_sample] += click.astype(np.float32)

        # Generate secondary click (grace note)
        secondary_time_ms = current_time_ms + secondary_delay_ms
        start_sample = int(secondary_time_ms * sample_rate / 1000)
        end_sample = min(start_sample + click_samples, total_samples)
        if end_sample > start_sample:
            t = np.arange(end_sample - start_sample) / sample_rate
            click = 0.5 * np.sin(2 * np.pi * 1200 * t) * np.exp(-t * 100)
            audio[start_sample:end_sample] += click.astype(np.float32)

        current_time_ms += interval_ms

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    signal_def = SignalDefinition(
        signal_type="onset",
        metadata=SignalMetadata(
            sample_rate=sample_rate,
            duration=duration,
            description=f"Double strikes (flam) with {secondary_delay_ms}ms delay",
        ),
        ground_truth=GroundTruth(onsets=onsets),
        test_criteria=TestCriteria(
            onset_timing_tolerance_ms=30.0,
            min_detection_rate=0.85,
        ),
    )

    return audio.astype(np.float32), signal_def


def _generate_mixed_attacks(
    duration: float, sample_rate: int = DEFAULT_SAMPLE_RATE
) -> tuple[np.ndarray, SignalDefinition]:
    """Generate signal with mixed attack types."""
    attack_types = ["impulse", "sharp", "medium", "slow"]
    interval_ms = 500.0

    total_samples = int(duration * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float32)
    onsets = []

    attack_times_ms = {
        "impulse": 0.1,
        "sharp": 1.0,
        "medium": 10.0,
        "slow": 50.0,
    }

    current_time_ms = 0.0
    idx = 0

    while current_time_ms < duration * 1000:
        attack_type = attack_types[idx % len(attack_types)]
        attack_ms = attack_times_ms[attack_type]

        onsets.append(
            OnsetAnnotation(
                time=current_time_ms / 1000.0,
                attack_ms=attack_ms,
                onset_type=attack_type,
            )
        )

        # Generate tone with appropriate attack
        tone_duration_ms = attack_ms + 100.0
        tone_samples = int(tone_duration_ms * sample_rate / 1000)
        start_sample = int(current_time_ms * sample_rate / 1000)
        end_sample = min(start_sample + tone_samples, total_samples)

        t = np.arange(end_sample - start_sample) / sample_rate
        attack_time = attack_ms / 1000.0

        envelope = np.minimum(t / attack_time if attack_time > 0 else 1.0, 1.0)
        envelope *= np.exp(-np.maximum(t - attack_time - 0.05, 0) * 20)

        tone = np.sin(2 * np.pi * 1000 * t) * envelope
        audio[start_sample:end_sample] += tone.astype(np.float32)

        current_time_ms += interval_ms
        idx += 1

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    signal_def = SignalDefinition(
        signal_type="onset",
        metadata=SignalMetadata(
            sample_rate=sample_rate,
            duration=duration,
            description="Mixed attack types: impulse, sharp, medium, slow",
        ),
        ground_truth=GroundTruth(onsets=onsets),
        test_criteria=TestCriteria(
            onset_timing_tolerance_ms=60.0,
            min_detection_rate=0.80,
        ),
    )

    return audio.astype(np.float32), signal_def


def _generate_sweep_onset(
    start_freq: float,
    end_freq: float,
    interval_ms: float,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, SignalDefinition]:
    """Generate onsets with frequency sweep."""
    total_samples = int(duration * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float32)
    onsets = []

    current_time_ms = 0.0
    num_onsets = int(duration * 1000 / interval_ms)

    for i in range(num_onsets):
        if current_time_ms >= duration * 1000:
            break

        # Interpolate frequency
        progress = i / max(num_onsets - 1, 1)
        freq = start_freq + (end_freq - start_freq) * progress

        onsets.append(
            OnsetAnnotation(
                time=current_time_ms / 1000.0,
                attack_ms=1.0,
                onset_type="sharp",
            )
        )

        # Generate tone at this frequency
        tone_duration_ms = 100.0
        tone_samples = int(tone_duration_ms * sample_rate / 1000)
        start_sample = int(current_time_ms * sample_rate / 1000)
        end_sample = min(start_sample + tone_samples, total_samples)

        t = np.arange(end_sample - start_sample) / sample_rate
        envelope = np.exp(-t * 30)
        tone = np.sin(2 * np.pi * freq * t) * envelope
        audio[start_sample:end_sample] += tone.astype(np.float32)

        current_time_ms += interval_ms

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    signal_def = SignalDefinition(
        signal_type="onset",
        metadata=SignalMetadata(
            sample_rate=sample_rate,
            duration=duration,
            description=f"Frequency sweep onsets: {start_freq}-{end_freq} Hz",
        ),
        ground_truth=GroundTruth(onsets=onsets),
        test_criteria=TestCriteria(
            onset_timing_tolerance_ms=50.0,
            min_detection_rate=0.85,
        ),
    )

    return audio.astype(np.float32), signal_def


def _generate_noisy_signal(
    bpm: float,
    snr_db: float,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    rng_seed: int = 42,
) -> tuple[np.ndarray, SignalDefinition]:
    """Generate click track with specified SNR noise."""
    audio, signal_def = generate_click_track(bpm=bpm, duration=duration)

    # Add noise
    signal_power = np.mean(audio**2)
    if signal_power > 0:
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise_amplitude = np.sqrt(noise_power)

        rng = np.random.default_rng(rng_seed)
        noise = noise_amplitude * rng.standard_normal(len(audio))
        audio = (audio + noise).astype(np.float32)

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = (audio / max_val * 0.9).astype(np.float32)

    signal_def.metadata.description = f"{bpm} BPM with {snr_db} dB SNR noise"
    signal_def.test_criteria.min_detection_rate = max(0.5, 0.95 - (20 - snr_db) * 0.02)

    return audio, signal_def


def _generate_clipped_signal(
    bpm: float,
    clip_threshold: float,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, SignalDefinition]:
    """Generate click track with clipped peaks."""
    audio, signal_def = generate_click_track(bpm=bpm, duration=duration)

    # Apply clipping
    audio = np.clip(audio, -clip_threshold, clip_threshold)

    signal_def.metadata.description = f"{bpm} BPM with clipping at {clip_threshold}"

    return audio.astype(np.float32), signal_def


def _generate_dc_offset_signal(
    bpm: float,
    dc_offset: float,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, SignalDefinition]:
    """Generate click track with DC offset."""
    audio, signal_def = generate_click_track(bpm=bpm, duration=duration)

    # Add DC offset
    audio = audio + dc_offset

    # Re-normalize to prevent clipping
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val * 0.9

    signal_def.metadata.description = f"{bpm} BPM with DC offset {dc_offset}"

    return audio.astype(np.float32), signal_def


def _generate_low_amplitude_signal(
    bpm: float,
    amplitude: float,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, SignalDefinition]:
    """Generate click track with very low amplitude."""
    audio, signal_def = generate_click_track(bpm=bpm, duration=duration)

    # Scale down to low amplitude
    audio = audio * amplitude

    signal_def.metadata.description = f"{bpm} BPM at {amplitude * 100}% amplitude"

    return audio.astype(np.float32), signal_def


def _generate_quantized_signal(
    bpm: float,
    bit_depth: int,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, SignalDefinition]:
    """Generate click track with low bit-depth quantization."""
    audio, signal_def = generate_click_track(bpm=bpm, duration=duration)

    # Quantize to specified bit depth
    levels = 2**bit_depth
    audio = np.round(audio * levels / 2) / (levels / 2)

    signal_def.metadata.description = f"{bpm} BPM quantized to {bit_depth}-bit"

    return audio.astype(np.float32), signal_def


def _generate_bandlimited_signal(
    bpm: float,
    low_freq: float,
    high_freq: float,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[np.ndarray, SignalDefinition]:
    """Generate bandlimited click track (simulating phone/low-quality audio)."""
    from scipy import signal as scipy_signal

    audio, signal_def = generate_click_track(bpm=bpm, duration=duration)

    # Design bandpass filter
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = min(high_freq / nyquist, 0.99)

    sos = scipy_signal.butter(4, [low, high], btype="band", output="sos")
    audio = scipy_signal.sosfilt(sos, audio).astype(np.float32)

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    signal_def.metadata.description = (
        f"{bpm} BPM bandlimited to {low_freq}-{high_freq} Hz"
    )

    return audio.astype(np.float32), signal_def
