"""Signal-Aware Analyzer Factory.

Provides intelligent analyzer selection and configuration based on
signal characteristics. Automatically selects optimal methods and
parameters for different types of audio content.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import signal as scipy_signal

from aubio_beatcheck.core.analyzers import (
    AnalyzerConfig,
    OnsetAnalyzer,
    PitchAnalyzer,
    TempoAnalyzer,
)


@dataclass
class SignalProfile:
    """Characteristics of an audio signal for analyzer selection.

    Computed from spectral and temporal analysis of the signal to
    determine optimal analyzer configuration.

    Attributes:
        spectral_centroid: Center of mass of the spectrum (Hz)
        spectral_flatness: Measure of noise-like quality (0-1)
        spectral_rolloff: Frequency below which 85% of energy lies
        attack_sharpness: Estimated attack time characteristic (0-1)
        energy_variance: Variance in energy envelope
        estimated_snr: Estimated signal-to-noise ratio (dB)
        zero_crossing_rate: Rate of sign changes per second
        peak_frequency: Dominant frequency in spectrum
        is_percussive: Whether signal appears percussive
        is_tonal: Whether signal appears tonal
    """

    spectral_centroid: float = 0.0
    spectral_flatness: float = 0.0
    spectral_rolloff: float = 0.0
    attack_sharpness: float = 0.5
    energy_variance: float = 0.0
    estimated_snr: float = 20.0
    zero_crossing_rate: float = 0.0
    peak_frequency: float = 0.0
    is_percussive: bool = False
    is_tonal: bool = False
    suggested_onset_method: str = "hfc"
    suggested_pitch_method: str = "yinfft"
    confidence: float = 0.5


class SignalClassifier:
    """Classify signal characteristics to select optimal analyzer.

    Analyzes audio signals to determine their characteristics and
    recommend optimal analyzer methods and parameters.
    """

    def __init__(self, sample_rate: int = 44100, frame_size: int = 2048):
        """Initialize signal classifier.

        Args:
            sample_rate: Audio sample rate in Hz
            frame_size: FFT frame size for analysis
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size

    def classify(self, audio: np.ndarray) -> SignalProfile:
        """Analyze signal to determine optimal analyzer configuration.

        Args:
            audio: Audio samples as float32 array

        Returns:
            SignalProfile with characteristics and recommendations
        """
        audio = np.asarray(audio, dtype=np.float32).flatten()

        if len(audio) < self.frame_size:
            # Signal too short for full analysis
            return SignalProfile(confidence=0.1)

        profile = SignalProfile()

        # Compute spectral features
        profile.spectral_centroid = self._compute_spectral_centroid(audio)
        profile.spectral_flatness = self._compute_spectral_flatness(audio)
        profile.spectral_rolloff = self._compute_spectral_rolloff(audio)
        profile.peak_frequency = self._find_peak_frequency(audio)

        # Compute temporal features
        profile.attack_sharpness = self._estimate_attack_sharpness(audio)
        profile.energy_variance = self._compute_energy_variance(audio)
        profile.zero_crossing_rate = self._compute_zcr(audio)

        # Estimate SNR
        profile.estimated_snr = self._estimate_snr(audio)

        # Classify signal type
        profile.is_percussive = self._is_percussive(profile)
        profile.is_tonal = self._is_tonal(profile)

        # Determine optimal methods
        profile.suggested_onset_method = self._suggest_onset_method(profile)
        profile.suggested_pitch_method = self._suggest_pitch_method(profile)
        profile.confidence = self._compute_confidence(profile)

        return profile

    def _compute_spectral_centroid(self, audio: np.ndarray) -> float:
        """Compute spectral centroid (center of mass of spectrum)."""
        # Use multiple frames and average
        num_frames = min(10, len(audio) // self.frame_size)
        if num_frames == 0:
            return 0.0

        centroids = []
        for i in range(num_frames):
            start = int(i * len(audio) / num_frames)
            frame = audio[start : start + self.frame_size]
            if len(frame) < self.frame_size:
                frame = np.pad(frame, (0, self.frame_size - len(frame)))

            spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
            freqs = np.fft.rfftfreq(len(frame), 1 / self.sample_rate)

            if np.sum(spectrum) > 0:
                centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
                centroids.append(centroid)

        return float(np.mean(centroids)) if centroids else 0.0

    def _compute_spectral_flatness(self, audio: np.ndarray) -> float:
        """Compute spectral flatness (measure of noise-like quality)."""
        frame = audio[: self.frame_size]
        if len(frame) < self.frame_size:
            frame = np.pad(frame, (0, self.frame_size - len(frame)))

        spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
        spectrum = spectrum + 1e-10  # Avoid log(0)

        geometric_mean = np.exp(np.mean(np.log(spectrum)))
        arithmetic_mean = np.mean(spectrum)

        if arithmetic_mean > 0:
            return float(geometric_mean / arithmetic_mean)
        return 0.0

    def _compute_spectral_rolloff(self, audio: np.ndarray, percentile: float = 0.85) -> float:
        """Compute spectral rolloff frequency."""
        frame = audio[: self.frame_size]
        if len(frame) < self.frame_size:
            frame = np.pad(frame, (0, self.frame_size - len(frame)))

        spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
        freqs = np.fft.rfftfreq(len(frame), 1 / self.sample_rate)

        cumsum = np.cumsum(spectrum)
        total_energy = cumsum[-1]

        if total_energy > 0:
            rolloff_idx = np.searchsorted(cumsum, percentile * total_energy)
            if rolloff_idx < len(freqs):
                return float(freqs[rolloff_idx])

        return float(self.sample_rate / 4)

    def _find_peak_frequency(self, audio: np.ndarray) -> float:
        """Find dominant frequency in spectrum."""
        frame = audio[: self.frame_size]
        if len(frame) < self.frame_size:
            frame = np.pad(frame, (0, self.frame_size - len(frame)))

        spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
        freqs = np.fft.rfftfreq(len(frame), 1 / self.sample_rate)

        # Ignore DC component
        spectrum[0] = 0
        peak_idx = np.argmax(spectrum)

        return float(freqs[peak_idx])

    def _estimate_attack_sharpness(self, audio: np.ndarray) -> float:
        """Estimate attack time characteristic (0=slow, 1=sharp)."""
        # Compute energy envelope
        frame_size = 256
        hop_size = 128
        envelope = []

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i : i + frame_size]
            envelope.append(np.sqrt(np.mean(frame**2)))

        if len(envelope) < 2:
            return 0.5

        envelope = np.array(envelope)

        # Find attack portions (energy increasing)
        diff = np.diff(envelope)
        attack_diffs = diff[diff > 0]

        if len(attack_diffs) == 0:
            return 0.5

        # Normalize attack speed (fast attack = high derivative)
        max_diff = np.max(attack_diffs)
        mean_diff = np.mean(attack_diffs)

        # Sharper attacks have higher max relative to mean
        if mean_diff > 0:
            sharpness = min(1.0, max_diff / (10 * mean_diff))
            return float(sharpness)

        return 0.5

    def _compute_energy_variance(self, audio: np.ndarray) -> float:
        """Compute variance in energy envelope."""
        frame_size = 512
        hop_size = 256
        envelope = []

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i : i + frame_size]
            envelope.append(np.sqrt(np.mean(frame**2)))

        if len(envelope) < 2:
            return 0.0

        return float(np.var(envelope))

    def _compute_zcr(self, audio: np.ndarray) -> float:
        """Compute zero crossing rate."""
        signs = np.sign(audio)
        signs[signs == 0] = 1  # Avoid zero values
        crossings = np.sum(np.abs(np.diff(signs))) / 2
        duration = len(audio) / self.sample_rate

        return float(crossings / duration) if duration > 0 else 0.0

    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio."""
        # Simple approach: compare peak to quiet sections
        frame_size = 1024
        hop_size = 512
        energies = []

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i : i + frame_size]
            energies.append(np.mean(frame**2))

        if len(energies) < 2:
            return 20.0

        energies = np.array(energies)
        signal_energy = np.percentile(energies, 90)
        noise_energy = np.percentile(energies, 10)

        if noise_energy > 0:
            snr = 10 * np.log10(signal_energy / noise_energy)
            return float(np.clip(snr, 0, 60))

        return 40.0

    def _is_percussive(self, profile: SignalProfile) -> bool:
        """Determine if signal is percussive."""
        # Percussive signals have:
        # - High energy variance
        # - Sharp attacks
        # - High spectral flatness (broadband)
        return (
            profile.attack_sharpness > 0.6
            and profile.spectral_flatness > 0.1
            and profile.energy_variance > 0.01
        )

    def _is_tonal(self, profile: SignalProfile) -> bool:
        """Determine if signal is tonal."""
        # Tonal signals have:
        # - Low spectral flatness (clear peaks)
        # - Lower ZCR relative to spectral centroid
        expected_zcr_for_freq = 2 * profile.peak_frequency
        zcr_ratio = (
            profile.zero_crossing_rate / expected_zcr_for_freq
            if expected_zcr_for_freq > 0
            else 1.0
        )

        return profile.spectral_flatness < 0.2 and 0.8 < zcr_ratio < 1.2

    def _suggest_onset_method(self, profile: SignalProfile) -> str:
        """Suggest optimal onset detection method based on profile."""
        if profile.is_percussive and profile.attack_sharpness > 0.7:
            return "hfc"  # High-frequency content for sharp percussive
        elif profile.spectral_flatness > 0.4:
            return "specflux"  # Better for noise-like/complex signals
        elif profile.is_tonal:
            return "complex"  # Phase-based for tonal signals
        elif profile.attack_sharpness < 0.3:
            return "mkl"  # Modified Kullback-Leibler for soft onsets
        else:
            return "hfc"  # Default to HFC

    def _suggest_pitch_method(self, profile: SignalProfile) -> str:
        """Suggest optimal pitch detection method based on profile."""
        if profile.is_tonal and profile.estimated_snr > 20:
            return "yinfft"  # Best for clean tonal signals
        elif profile.estimated_snr < 10:
            return "fcomb"  # More noise robust
        elif profile.spectral_flatness > 0.3:
            return "mcomb"  # Multi-comb for complex spectra
        else:
            return "yinfft"  # Default to YIN FFT

    def _compute_confidence(self, profile: SignalProfile) -> float:
        """Compute confidence in the classification."""
        # Higher confidence for clearer signal characteristics
        confidence = 0.5

        if profile.estimated_snr > 30:
            confidence += 0.2
        elif profile.estimated_snr > 15:
            confidence += 0.1

        if profile.is_percussive or profile.is_tonal:
            confidence += 0.2

        if profile.spectral_flatness < 0.1 or profile.spectral_flatness > 0.5:
            confidence += 0.1

        return min(1.0, confidence)


class AnalyzerFactory:
    """Factory that creates optimally-configured analyzers based on signal profile.

    Provides intelligent analyzer instantiation by analyzing the input
    signal and selecting appropriate methods and parameters.
    """

    def __init__(self, sample_rate: int = 44100):
        """Initialize analyzer factory.

        Args:
            sample_rate: Default sample rate for analyzers
        """
        self.sample_rate = sample_rate
        self.classifier = SignalClassifier(sample_rate=sample_rate)

    def create_tempo_analyzer(
        self,
        audio: np.ndarray | None = None,
        profile: SignalProfile | None = None,
        config: AnalyzerConfig | None = None,
    ) -> TempoAnalyzer:
        """Create tempo analyzer optimized for the signal.

        Args:
            audio: Optional audio to analyze for configuration
            profile: Optional pre-computed signal profile
            config: Optional explicit configuration (overrides auto)

        Returns:
            Configured TempoAnalyzer
        """
        if config is not None:
            return TempoAnalyzer(config)

        if profile is None and audio is not None:
            profile = self.classifier.classify(audio)

        # Adjust FFT size based on expected tempo range
        if profile and profile.is_percussive:
            # Shorter FFT for percussive signals (better time resolution)
            fft_size = 1024
            hop_size = 256
        else:
            # Standard FFT for general signals
            fft_size = 2048
            hop_size = 512

        analyzer_config = AnalyzerConfig(
            fft_size=fft_size,
            hop_size=hop_size,
            sample_rate=self.sample_rate,
        )

        return TempoAnalyzer(analyzer_config)

    def create_onset_analyzer(
        self,
        audio: np.ndarray | None = None,
        profile: SignalProfile | None = None,
        config: AnalyzerConfig | None = None,
        method: str | None = None,
        threshold: float | None = None,
    ) -> OnsetAnalyzer:
        """Create onset analyzer optimized for the signal.

        Args:
            audio: Optional audio to analyze for configuration
            profile: Optional pre-computed signal profile
            config: Optional explicit configuration
            method: Optional explicit method override
            threshold: Optional explicit threshold override

        Returns:
            Configured OnsetAnalyzer
        """
        if profile is None and audio is not None:
            profile = self.classifier.classify(audio)

        # Determine method
        if method is None:
            method = profile.suggested_onset_method if profile else "hfc"

        # Determine threshold based on signal characteristics
        if threshold is None:
            threshold = self._compute_onset_threshold(profile)

        # Determine FFT configuration
        if config is None:
            if profile and profile.attack_sharpness > 0.7:
                # Shorter FFT for sharp attacks
                config = AnalyzerConfig(
                    fft_size=1024, hop_size=256, sample_rate=self.sample_rate
                )
            elif profile and profile.attack_sharpness < 0.3:
                # Longer FFT for gradual attacks
                config = AnalyzerConfig(
                    fft_size=4096, hop_size=512, sample_rate=self.sample_rate
                )
            else:
                config = AnalyzerConfig(sample_rate=self.sample_rate)

        return OnsetAnalyzer(config, method=method, threshold=threshold)

    def create_pitch_analyzer(
        self,
        audio: np.ndarray | None = None,
        profile: SignalProfile | None = None,
        config: AnalyzerConfig | None = None,
        method: str | None = None,
        tolerance: float | None = None,
    ) -> PitchAnalyzer:
        """Create pitch analyzer optimized for the signal.

        Args:
            audio: Optional audio to analyze for configuration
            profile: Optional pre-computed signal profile
            config: Optional explicit configuration
            method: Optional explicit method override
            tolerance: Optional explicit tolerance override

        Returns:
            Configured PitchAnalyzer
        """
        if profile is None and audio is not None:
            profile = self.classifier.classify(audio)

        # Determine method
        if method is None:
            method = profile.suggested_pitch_method if profile else "yinfft"

        # Determine tolerance
        if tolerance is None:
            tolerance = self._compute_pitch_tolerance(profile)

        # Determine FFT configuration based on frequency range
        if config is None:
            if profile and profile.peak_frequency < 200:
                # Longer FFT for low frequencies
                config = AnalyzerConfig(
                    fft_size=4096, hop_size=512, sample_rate=self.sample_rate
                )
            else:
                config = AnalyzerConfig(sample_rate=self.sample_rate)

        return PitchAnalyzer(config, method=method, tolerance=tolerance)

    def create_all_analyzers(
        self, audio: np.ndarray
    ) -> tuple[TempoAnalyzer, OnsetAnalyzer, PitchAnalyzer]:
        """Create all analyzers optimized for the same signal.

        Args:
            audio: Audio to analyze

        Returns:
            Tuple of (TempoAnalyzer, OnsetAnalyzer, PitchAnalyzer)
        """
        profile = self.classifier.classify(audio)

        return (
            self.create_tempo_analyzer(profile=profile),
            self.create_onset_analyzer(profile=profile),
            self.create_pitch_analyzer(profile=profile),
        )

    def _compute_onset_threshold(self, profile: SignalProfile | None) -> float:
        """Compute optimal onset threshold based on profile."""
        if profile is None:
            return 0.3

        # Lower threshold for noisy signals (more sensitive)
        if profile.estimated_snr < 15:
            base_threshold = 0.2
        elif profile.estimated_snr > 30:
            base_threshold = 0.4
        else:
            base_threshold = 0.3

        # Adjust for attack sharpness
        if profile.attack_sharpness > 0.7:
            # Sharp attacks are easy to detect
            return min(0.5, base_threshold + 0.1)
        elif profile.attack_sharpness < 0.3:
            # Soft attacks need lower threshold
            return max(0.1, base_threshold - 0.1)

        return base_threshold

    def _compute_pitch_tolerance(self, profile: SignalProfile | None) -> float:
        """Compute optimal pitch tolerance based on profile."""
        if profile is None:
            return 0.8

        # Lower tolerance (stricter) for clean tonal signals
        if profile.is_tonal and profile.estimated_snr > 30:
            return 0.9
        elif profile.estimated_snr < 15:
            return 0.6
        else:
            return 0.8


class AnalyzerSelector:
    """Convenience class for quick analyzer method selection.

    Provides static methods to quickly determine optimal methods
    without full signal analysis.
    """

    # Method recommendations by signal type
    ONSET_METHODS = {
        "percussive": "hfc",
        "tonal": "complex",
        "noisy": "specflux",
        "soft": "mkl",
        "default": "hfc",
    }

    PITCH_METHODS = {
        "clean": "yinfft",
        "noisy": "fcomb",
        "complex": "mcomb",
        "default": "yinfft",
    }

    @classmethod
    def recommend_onset_method(
        cls, signal_type: Literal["percussive", "tonal", "noisy", "soft", "default"]
    ) -> str:
        """Get recommended onset method for signal type.

        Args:
            signal_type: Type of signal

        Returns:
            Recommended onset detection method
        """
        return cls.ONSET_METHODS.get(signal_type, cls.ONSET_METHODS["default"])

    @classmethod
    def recommend_pitch_method(
        cls, signal_type: Literal["clean", "noisy", "complex", "default"]
    ) -> str:
        """Get recommended pitch method for signal type.

        Args:
            signal_type: Type of signal

        Returns:
            Recommended pitch detection method
        """
        return cls.PITCH_METHODS.get(signal_type, cls.PITCH_METHODS["default"])

    @classmethod
    def list_onset_methods(cls) -> list[str]:
        """List all available onset detection methods."""
        return ["energy", "hfc", "complex", "phase", "specdiff", "kl", "mkl", "specflux"]

    @classmethod
    def list_pitch_methods(cls) -> list[str]:
        """List all available pitch detection methods."""
        return ["default", "schmitt", "fcomb", "mcomb", "yin", "yinfast", "yinfft"]
