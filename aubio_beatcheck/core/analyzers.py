"""
Aubio Analysis Wrappers

This module provides clean wrapper classes for aubio's tempo, onset, pitch,
and phase vocoder analysis functions. These classes encapsulate configuration,
frame-by-frame processing, and performance metrics collection.
"""

import time
from dataclasses import dataclass, field
from typing import Literal

import aubio
import numpy as np


@dataclass
class AnalyzerConfig:
    """Configuration for an aubio analyzer."""

    fft_size: int = 2048
    hop_size: int = 512
    sample_rate: int = 44100


@dataclass
class PerformanceStats:
    """Performance statistics for analyzer."""

    frame_times_us: list[float] = field(default_factory=list)

    @property
    def mean_us(self) -> float:
        return float(np.mean(self.frame_times_us)) if self.frame_times_us else 0.0

    @property
    def p95_us(self) -> float:
        return (
            float(np.percentile(self.frame_times_us, 95))
            if self.frame_times_us
            else 0.0
        )

    @property
    def p99_us(self) -> float:
        return (
            float(np.percentile(self.frame_times_us, 99))
            if self.frame_times_us
            else 0.0
        )

    @property
    def max_us(self) -> float:
        return float(np.max(self.frame_times_us)) if self.frame_times_us else 0.0


class TempoAnalyzer:
    """
    Wrapper for aubio tempo detection.

    Provides beat tracking and BPM estimation with configurable FFT parameters.
    """

    def __init__(
        self,
        config: AnalyzerConfig | None = None,
        method: str = "default",
        enable_features: bool = True,
    ):
        """
        Initialize tempo analyzer.

        Args:
            config: FFT configuration (uses defaults if None)
            method: Aubio tempo method (only 'default' is typically supported)
            enable_features: Whether to enable advanced tempo features
        """
        self.config = config or AnalyzerConfig()
        self.method = method
        self.stats = PerformanceStats()

        self._tempo = aubio.tempo(
            method,
            self.config.fft_size,
            self.config.hop_size,
            self.config.sample_rate,
        )

        self.enabled_features: list[str] = []
        if enable_features:
            self._enable_advanced_features()

    def _enable_advanced_features(self):
        """Enable advanced tempo features if available in aubio build."""
        features = [
            ("multi_octave", lambda: self._tempo.set_multi_octave(1)),
            ("onset_enhancement", lambda: self._tempo.set_onset_enhancement(1)),
            ("fft_autocorr", lambda: self._tempo.set_fft_autocorr(1)),
            ("dynamic_tempo", lambda: self._tempo.set_dynamic_tempo(1)),
            ("adaptive_winlen", lambda: self._tempo.set_adaptive_winlen(1)),
            ("use_tempogram", lambda: self._tempo.set_use_tempogram(1)),
        ]

        for name, setter in features:
            try:
                setter()
                self.enabled_features.append(name)
            except (ValueError, RuntimeError, AttributeError):
                pass  # Feature not available

    def analyze(self, audio: np.ndarray) -> tuple[list[float], float]:
        """
        Analyze tempo and beats in audio signal.

        Args:
            audio: Audio samples as float32 array

        Returns:
            Tuple of (beat_times_seconds, detected_bpm)
        """
        # Ensure audio is float32 and contiguous
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        detected_beats = []
        hop_size = self.config.hop_size

        for i in range(0, len(audio), hop_size):
            chunk = audio[i : i + hop_size]

            # Ensure chunk is exactly hop_size samples
            if len(chunk) < hop_size:
                chunk = np.pad(chunk, (0, hop_size - len(chunk)), mode="constant")
            elif len(chunk) > hop_size:
                chunk = chunk[:hop_size]

            t0 = time.perf_counter()
            is_beat = self._tempo(chunk)
            elapsed_us = (time.perf_counter() - t0) * 1_000_000
            self.stats.frame_times_us.append(elapsed_us)

            if is_beat:
                beat_time = i / self.config.sample_rate
                detected_beats.append(beat_time)

        detected_bpm = float(self._tempo.get_bpm())
        return detected_beats, detected_bpm

    def reset(self):
        """Reset analyzer state and statistics."""
        self.stats = PerformanceStats()
        # Note: aubio tempo doesn't have a reset method, so we recreate
        self._tempo = aubio.tempo(
            self.method,
            self.config.fft_size,
            self.config.hop_size,
            self.config.sample_rate,
        )
        if self.enabled_features:
            self._enable_advanced_features()


class OnsetAnalyzer:
    """
    Wrapper for aubio onset detection.

    Detects transient events (note onsets, percussive hits, etc.).
    """

    def __init__(
        self,
        config: AnalyzerConfig | None = None,
        method: Literal[
            "energy", "hfc", "complex", "phase", "specdiff", "kl", "mkl", "specflux"
        ] = "hfc",
        threshold: float = 0.3,
    ):
        """
        Initialize onset analyzer.

        Args:
            config: FFT configuration (uses defaults if None)
            method: Onset detection method
            threshold: Detection threshold (0.0-1.0)
        """
        self.config = config or AnalyzerConfig()
        self.method = method
        self.stats = PerformanceStats()

        self._onset = aubio.onset(
            method,
            self.config.fft_size,
            self.config.hop_size,
            self.config.sample_rate,
        )
        self._onset.set_threshold(threshold)

    def analyze(self, audio: np.ndarray) -> list[float]:
        """
        Detect onsets in audio signal.

        Args:
            audio: Audio samples as float32 array

        Returns:
            List of onset times in seconds
        """
        # Ensure audio is float32 and contiguous
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        detected_onsets = []
        hop_size = self.config.hop_size

        for i in range(0, len(audio), hop_size):
            chunk = audio[i : i + hop_size]

            # Ensure chunk is exactly hop_size samples
            if len(chunk) < hop_size:
                chunk = np.pad(chunk, (0, hop_size - len(chunk)), mode="constant")
            elif len(chunk) > hop_size:
                chunk = chunk[:hop_size]

            t0 = time.perf_counter()
            is_onset = self._onset(chunk)
            elapsed_us = (time.perf_counter() - t0) * 1_000_000
            self.stats.frame_times_us.append(elapsed_us)

            if is_onset:
                onset_time = i / self.config.sample_rate
                detected_onsets.append(onset_time)

        return detected_onsets

    def set_threshold(self, threshold: float):
        """Update onset detection threshold."""
        self._onset.set_threshold(threshold)

    def reset(self):
        """Reset analyzer state and statistics."""
        self.stats = PerformanceStats()
        self._onset = aubio.onset(
            self.method,
            self.config.fft_size,
            self.config.hop_size,
            self.config.sample_rate,
        )


class PitchAnalyzer:
    """
    Wrapper for aubio pitch detection.

    Detects fundamental frequency and converts to MIDI note numbers.
    """

    def __init__(
        self,
        config: AnalyzerConfig | None = None,
        method: Literal[
            "default", "schmitt", "fcomb", "mcomb", "yin", "yinfast", "yinfft"
        ] = "yinfft",
        tolerance: float = 0.8,
    ):
        """
        Initialize pitch analyzer.

        Args:
            config: FFT configuration (uses defaults if None)
            method: Pitch detection method
            tolerance: Tolerance parameter (method-specific)
        """
        self.config = config or AnalyzerConfig()
        self.method = method
        self.stats = PerformanceStats()

        self._pitch = aubio.pitch(
            method,
            self.config.fft_size,
            self.config.hop_size,
            self.config.sample_rate,
        )
        self._pitch.set_unit("midi")
        self._pitch.set_tolerance(tolerance)

    def analyze(
        self, audio: np.ndarray, min_confidence: float = 0.0
    ) -> list[tuple[float, float, float]]:
        """
        Detect pitch in audio signal.

        Args:
            audio: Audio samples as float32 array
            min_confidence: Minimum confidence threshold (0.0-1.0)

        Returns:
            List of (time_seconds, midi_note, confidence) tuples
        """
        # Ensure audio is float32 and contiguous
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        detected_pitches = []
        hop_size = self.config.hop_size

        for i in range(0, len(audio), hop_size):
            chunk = audio[i : i + hop_size]

            # Ensure chunk is exactly hop_size samples
            if len(chunk) < hop_size:
                chunk = np.pad(chunk, (0, hop_size - len(chunk)), mode="constant")
            elif len(chunk) > hop_size:
                chunk = chunk[:hop_size]

            t0 = time.perf_counter()
            midi_note = float(self._pitch(chunk)[0])
            elapsed_us = (time.perf_counter() - t0) * 1_000_000
            self.stats.frame_times_us.append(elapsed_us)

            confidence = float(self._pitch.get_confidence())
            pitch_time = i / self.config.sample_rate

            # Filter valid detections (MIDI 21 = A0, lowest piano note)
            if midi_note > 20 and confidence >= min_confidence:
                detected_pitches.append((pitch_time, midi_note, confidence))

        return detected_pitches

    def set_tolerance(self, tolerance: float):
        """Update pitch detection tolerance."""
        self._pitch.set_tolerance(tolerance)

    def reset(self):
        """Reset analyzer state and statistics."""
        self.stats = PerformanceStats()
        self._pitch = aubio.pitch(
            self.method,
            self.config.fft_size,
            self.config.hop_size,
            self.config.sample_rate,
        )
        self._pitch.set_unit("midi")


class PvocAnalyzer:
    """
    Wrapper for aubio phase vocoder.

    Performs time-stretching and pitch-shifting operations.
    """

    def __init__(
        self,
        config: AnalyzerConfig | None = None,
    ):
        """
        Initialize phase vocoder analyzer.

        Args:
            config: FFT configuration (uses defaults if None)
        """
        self.config = config or AnalyzerConfig()
        self.stats = PerformanceStats()

        self._pvoc = aubio.pvoc(self.config.fft_size, self.config.hop_size)

    def analyze_forward(self, audio: np.ndarray) -> np.ndarray:
        """
        Perform forward phase vocoder transform (time -> frequency).

        Args:
            audio: Audio samples as float32 array

        Returns:
            Complex spectrum (cvec)
        """
        # Ensure audio is float32 and contiguous
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        hop_size = self.config.hop_size
        spectra = []

        for i in range(0, len(audio), hop_size):
            chunk = audio[i : i + hop_size]

            # Ensure chunk is exactly hop_size samples
            if len(chunk) < hop_size:
                chunk = np.pad(chunk, (0, hop_size - len(chunk)), mode="constant")
            elif len(chunk) > hop_size:
                chunk = chunk[:hop_size]

            t0 = time.perf_counter()
            spectrum = self._pvoc(chunk)
            elapsed_us = (time.perf_counter() - t0) * 1_000_000
            self.stats.frame_times_us.append(elapsed_us)

            spectra.append(spectrum)

        return np.array(spectra)

    def analyze_inverse(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Perform inverse phase vocoder transform (frequency -> time).

        Args:
            spectrum: Complex spectrum

        Returns:
            Audio samples
        """
        # Note: Inverse pvoc requires different handling
        # This is a placeholder for the interface
        raise NotImplementedError("Inverse pvoc not yet implemented")

    def reset(self):
        """Reset analyzer state and statistics."""
        self.stats = PerformanceStats()
        self._pvoc = aubio.pvoc(self.config.fft_size, self.config.hop_size)
