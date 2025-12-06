"""TheBeat-Based Signal Generator.

Generates audio signals using the thebeat library with authoritative ground truth timing.
"""

import numpy as np
from thebeat import Sequence, SoundSequence, SoundStimulus

from .ground_truth import (
    BeatAnnotation,
    GroundTruth,
    OnsetAnnotation,
    PitchAnnotation,
    SignalDefinition,
    SignalMetadata,
    TestCriteria,
)

DEFAULT_SAMPLE_RATE = 44100


def midi_to_frequency(midi_note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def round_to_sample_boundary(seq: Sequence, sample_rate: int) -> None:
    """Round sequence onsets to exact sample boundaries.

    This eliminates the thebeat frame rounding warning by ensuring that
    onset times in milliseconds correspond to exact integer sample positions.

    Args:
        seq: The Sequence object to modify in place.
        sample_rate: Audio sample rate in Hz.
    """
    # Sample duration in ms = 1000 / sample_rate
    # For 44100 Hz, this is ~0.0227 ms per sample
    sample_duration_ms = 1000.0 / sample_rate

    # Get current onsets and round to nearest sample boundary
    current_onsets = seq.onsets.copy()
    
    # Convert to samples, round to integer, convert back to ms
    samples = current_onsets * sample_rate / 1000.0
    aligned_samples = np.round(samples)
    aligned_onsets = aligned_samples * 1000.0 / sample_rate

    # Update the sequence's onsets using the property setter
    seq.onsets = aligned_onsets


def align_duration_to_samples(duration_ms: float, sample_rate: int) -> float:
    """Align a duration in milliseconds to exact sample boundaries.

    Ensures the duration corresponds to an integer number of samples.

    Args:
        duration_ms: Duration in milliseconds.
        sample_rate: Audio sample rate in Hz.

    Returns:
        Aligned duration in milliseconds.
    """
    samples = duration_ms * sample_rate / 1000.0
    aligned_samples = round(samples)
    return aligned_samples * 1000.0 / sample_rate


def extract_ground_truth_from_sequence(
    trial: SoundSequence,
    signal_type: str = "tempo",
    midi_notes: list[int] | None = None,
    attack_type: str = "sharp",
    attack_ms: float = 1.0,
) -> GroundTruth:
    """Extract ground truth data from a thebeat SoundSequence."""
    ground_truth = GroundTruth()
    onsets_seconds = trial.onsets / 1000.0

    if signal_type in ("tempo", "complex"):
        ground_truth.beats = [
            BeatAnnotation(
                time=float(onset),
                beat_number=(i % 4) + 1,
                bar=(i // 4) + 1,
            )
            for i, onset in enumerate(onsets_seconds)
        ]

    if signal_type == "onset":
        ground_truth.onsets = [
            OnsetAnnotation(
                time=float(onset),
                attack_ms=attack_ms,
                onset_type=attack_type,
            )
            for onset in onsets_seconds
        ]

    if signal_type == "pitch":
        if midi_notes is None:
            raise ValueError("midi_notes required for pitch signal ground truth")

        if len(midi_notes) != len(onsets_seconds):
            raise ValueError(
                f"MIDI notes length ({len(midi_notes)}) must match onsets ({len(onsets_seconds)})"
            )

        iois_seconds = trial.iois / 1000.0
        ground_truth.pitches = []

        for i, (onset, midi_note) in enumerate(
            zip(onsets_seconds, midi_notes, strict=False)
        ):
            if i < len(iois_seconds):
                duration = float(iois_seconds[i])
            else:
                duration = float(iois_seconds[-1]) if len(iois_seconds) > 0 else 1.0

            ground_truth.pitches.append(
                PitchAnnotation(
                    start_time=float(onset),
                    end_time=float(onset + duration),
                    midi_note=midi_note,
                    frequency_hz=midi_to_frequency(midi_note),
                    waveform="sine",
                )
            )

    return ground_truth


def generate_click_track(
    bpm: float,
    duration: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    click_duration_ms: float = 50.0,
    click_freq: float = 1000.0,
    add_timing_jitter: bool = False,
    jitter_std_ms: float = 0.0,
    rng_seed: int | None = None,
) -> tuple[np.ndarray, SignalDefinition]:
    """Generate a click track at a specific tempo using thebeat."""
    from loguru import logger

    ioi_ms = 60000.0 / bpm
    n_events = int(duration / (ioi_ms / 1000.0))

    # Create sequence first to get thebeat's exact timing
    seq = Sequence.generate_isochronous(
        n_events=n_events, ioi=ioi_ms, end_with_interval=False
    )

    if add_timing_jitter and jitter_std_ms > 0:
        rng = np.random.default_rng(rng_seed)
        seq.add_noise_gaussian(noise_sd=jitter_std_ms, rng=rng)

    round_to_sample_boundary(seq, sample_rate)

    # Calculate click duration based on IOI to ensure it fits
    # thebeat's SoundSequence will place the sound at specific sample positions
    # We need to ensure the click is short enough to fit
    if len(seq.iois) > 0:
        min_ioi_ms = float(np.min(seq.iois))
        # Use 10% of the shortest IOI, capped at requested click_duration_ms
        safe_click_duration_ms = min(click_duration_ms, min_ioi_ms * 0.1)
    else:
        safe_click_duration_ms = click_duration_ms

    # Calculate exact samples using round() to match thebeat's calculation
    # thebeat's synthesize_sound uses: round(fs * t) where t = duration_ms / 1000
    actual_click_samples = round(safe_click_duration_ms * sample_rate / 1000.0)

    # Ensure even number of samples to avoid rounding issues in thebeat's SoundSequence
    # when placing sounds at half-sample positions (e.g. 0.5 rounds to 0, 1.5 rounds to 2)
    if actual_click_samples % 2 != 0:
        actual_click_samples -= 1

    # Convert back to ms - this value when multiplied by fs/1000 should give exact integer
    actual_click_duration_ms = (actual_click_samples * 1000.0) / sample_rate

    logger.debug(
        f"BPM={bpm}, IOI={ioi_ms:.2f}ms, min_ioi={min_ioi_ms if len(seq.iois) > 0 else 'N/A'}ms, "
        f"requested_click={click_duration_ms}ms, safe_click={safe_click_duration_ms:.4f}ms, "
        f"actual_click={actual_click_duration_ms:.4f}ms ({actual_click_samples} samples)"
    )

    # Align ramp times to sample boundaries
    aligned_ramp_ms = align_duration_to_samples(5.0, sample_rate)

    stim = SoundStimulus.generate(
        freq=click_freq,
        duration_ms=actual_click_duration_ms,
        fs=sample_rate,
        onramp_ms=aligned_ramp_ms,
        offramp_ms=aligned_ramp_ms,
        ramp_type="raised-cosine",
        amplitude=1.0,
        oscillator="sine",
    )

    logger.debug(
        f"SoundStimulus created: duration_ms={stim.duration_ms}, n_samples={len(stim.samples)} "
        f"(expected {actual_click_samples} samples)"
    )

    trial = SoundSequence(stim, seq, sequence_time_unit="ms")
    ground_truth = extract_ground_truth_from_sequence(trial, signal_type="tempo")
    actual_bpm = 60000.0 / np.mean(trial.iois) if len(trial.iois) > 0 else bpm

    signal_def = SignalDefinition(
        signal_type="tempo",
        metadata=SignalMetadata(
            sample_rate=sample_rate,
            duration=trial.duration / 1000.0,
            description=f"Click track at {bpm} BPM (thebeat-generated)",
            bpm=actual_bpm,
        ),
        ground_truth=ground_truth,
        test_criteria=TestCriteria(
            tempo_tolerance_bpm=2.0,
            beat_timing_tolerance_ms=50.0,
            min_detection_rate=0.95,
        ),
    )

    return trial.samples.astype(np.float32), signal_def


def generate_onset_signal(
    attack_type: str,
    interval_ms: float = 500.0,
    duration: float = 10.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frequency: float = 1000.0,
    waveform: str = "sine",
) -> tuple[np.ndarray, SignalDefinition]:
    """Generate onset test signals with various attack characteristics."""
    attack_times = {
        "impulse": 0.1,
        "sharp": 1.0,
        "medium": 10.0,
        "slow": 50.0,
    }

    if attack_type not in attack_times:
        raise ValueError(
            f"Unknown attack type: {attack_type}. Must be one of {list(attack_times.keys())}"
        )

    attack_ms = attack_times[attack_type]
    onset_duration_ms = attack_ms + 100.0

    # Align all durations to sample boundaries
    aligned_interval_ms = align_duration_to_samples(interval_ms, sample_rate)
    aligned_duration_ms = align_duration_to_samples(onset_duration_ms, sample_rate)
    aligned_attack_ms = align_duration_to_samples(attack_ms, sample_rate)
    aligned_offramp_ms = align_duration_to_samples(50.0, sample_rate)

    stim = SoundStimulus.generate(
        freq=frequency,
        duration_ms=aligned_duration_ms,
        fs=sample_rate,
        onramp_ms=aligned_attack_ms,
        offramp_ms=aligned_offramp_ms,
        ramp_type="linear" if attack_type != "impulse" else "raised-cosine",
        amplitude=1.0,
        oscillator="sine",
    )

    n_events = int(duration / (aligned_interval_ms / 1000.0))
    seq = Sequence.generate_isochronous(
        n_events=n_events, ioi=aligned_interval_ms, end_with_interval=False
    )
    round_to_sample_boundary(seq, sample_rate)

    trial = SoundSequence(stim, seq, sequence_time_unit="ms")
    ground_truth = extract_ground_truth_from_sequence(
        trial, signal_type="onset", attack_type=attack_type, attack_ms=aligned_attack_ms
    )

    signal_def = SignalDefinition(
        signal_type="onset",
        metadata=SignalMetadata(
            sample_rate=sample_rate,
            duration=trial.duration / 1000.0,
            description=f"Onset signal with {attack_type} attacks at {interval_ms}ms intervals (thebeat)",
        ),
        ground_truth=ground_truth,
        test_criteria=TestCriteria(
            onset_timing_tolerance_ms=50.0, min_detection_rate=0.90
        ),
    )

    audio = trial.samples.astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    return audio, signal_def


def generate_pitch_sequence(
    midi_notes: list[int],
    note_duration: float = 1.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    waveform: str = "sine",
) -> tuple[np.ndarray, SignalDefinition]:
    """Generate a sequence of pitched tones with ground truth."""
    oscillator_map = {
        "sine": "sine",
        "sawtooth": "sawtooth",
        "square": "square",
        "triangle": "sine",
    }
    oscillator = oscillator_map.get(waveform, "sine")

    note_duration_ms = note_duration * 1000.0
    fade_ms = 10.0

    # Align durations to sample boundaries
    aligned_note_duration_ms = align_duration_to_samples(note_duration_ms, sample_rate)
    aligned_fade_ms = align_duration_to_samples(fade_ms, sample_rate)

    stimuli = []
    for midi_note in midi_notes:
        freq = midi_to_frequency(midi_note)
        stim = SoundStimulus.generate(
            freq=freq,
            duration_ms=aligned_note_duration_ms,
            fs=sample_rate,
            onramp_ms=aligned_fade_ms,
            offramp_ms=aligned_fade_ms,
            amplitude=0.8,
            oscillator=oscillator,
        )
        stimuli.append(stim)

    seq = Sequence.generate_isochronous(
        n_events=len(midi_notes), ioi=aligned_note_duration_ms, end_with_interval=False
    )
    round_to_sample_boundary(seq, sample_rate)

    trial = SoundSequence(stimuli, seq, sequence_time_unit="ms")
    ground_truth = extract_ground_truth_from_sequence(
        trial, signal_type="pitch", midi_notes=midi_notes
    )

    for pitch_annot in ground_truth.pitches:
        pitch_annot.waveform = waveform

    total_duration = trial.duration / 1000.0
    signal_def = SignalDefinition(
        signal_type="pitch",
        metadata=SignalMetadata(
            sample_rate=sample_rate,
            duration=total_duration,
            description=f"Pitch sequence: {len(midi_notes)} notes, {waveform} waveform (thebeat)",
        ),
        ground_truth=ground_truth,
        test_criteria=TestCriteria(pitch_tolerance_cents=50.0, min_detection_rate=0.90),
    )

    return trial.samples.astype(np.float32), signal_def


def generate_chromatic_scale(
    start_midi: int = 60,
    num_notes: int = 13,
    note_duration: float = 0.5,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    waveform: str = "sine",
) -> tuple[np.ndarray, SignalDefinition]:
    """Generate a chromatic scale for pitch detection testing."""
    midi_notes = list(range(start_midi, start_midi + num_notes))
    return generate_pitch_sequence(
        midi_notes=midi_notes,
        note_duration=note_duration,
        sample_rate=sample_rate,
        waveform=waveform,
    )


def generate_rhythmic_pattern(
    pattern: str,
    grid_ioi_ms: float,
    click_duration_ms: float = 50.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frequency: float = 1000.0,
) -> tuple[np.ndarray, SignalDefinition]:
    """Generate a rhythmic pattern from binary string notation."""
    # Align all durations to sample boundaries
    aligned_click_ms = align_duration_to_samples(click_duration_ms, sample_rate)
    aligned_grid_ms = align_duration_to_samples(grid_ioi_ms, sample_rate)
    aligned_ramp_ms = align_duration_to_samples(5.0, sample_rate)

    stim = SoundStimulus.generate(
        freq=frequency,
        duration_ms=aligned_click_ms,
        fs=sample_rate,
        onramp_ms=aligned_ramp_ms,
        offramp_ms=aligned_ramp_ms,
        amplitude=1.0,
        oscillator="sine",
    )

    seq = Sequence.from_binary_string(pattern, grid_ioi=aligned_grid_ms)
    round_to_sample_boundary(seq, sample_rate)

    trial = SoundSequence(stim, seq, sequence_time_unit="ms")
    ground_truth = extract_ground_truth_from_sequence(trial, signal_type="tempo")

    avg_ioi_ms = np.mean(trial.iois) if len(trial.iois) > 0 else aligned_grid_ms
    avg_bpm = 60000.0 / avg_ioi_ms

    signal_def = SignalDefinition(
        signal_type="tempo",
        metadata=SignalMetadata(
            sample_rate=sample_rate,
            duration=trial.duration / 1000.0,
            description=f"Rhythmic pattern '{pattern}' at {grid_ioi_ms}ms grid (thebeat)",
            bpm=avg_bpm,
        ),
        ground_truth=ground_truth,
        test_criteria=TestCriteria(
            tempo_tolerance_bpm=5.0,
            beat_timing_tolerance_ms=50.0,
            min_detection_rate=0.85,
        ),
    )

    return trial.samples.astype(np.float32), signal_def


def generate_complex_signal(
    bpm: float = 120,
    duration: float = 30.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    melody_notes: list[int] | None = None,
    snr_db: float = 20.0,
    rng_seed: int | None = None,
) -> tuple[np.ndarray, SignalDefinition]:
    """Generate a complex test signal combining beats and pitch with noise."""
    audio, signal_def = generate_click_track(
        bpm=bpm, duration=duration, sample_rate=sample_rate
    )

    if melody_notes is None:
        melody_notes = [60, 62, 64, 65, 67, 69, 71, 72]

    note_duration = duration / len(melody_notes)
    melody_audio, _ = generate_pitch_sequence(
        midi_notes=melody_notes,
        note_duration=note_duration,
        sample_rate=sample_rate,
        waveform="sine",
    )

    min_len = min(len(audio), len(melody_audio))
    audio[:min_len] += 0.3 * melody_audio[:min_len]

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

    signal_def.signal_type = "complex"
    signal_def.metadata.description = (
        f"Complex signal: {bpm} BPM click + melody + {snr_db}dB SNR noise (thebeat)"
    )

    return audio, signal_def
