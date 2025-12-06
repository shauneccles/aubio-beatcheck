"""Ground Truth Schema for Audio Analysis Testing.

This module defines Pydantic models for ground truth data used in
validating audio analysis accuracy. Ground truth accompanies each
synthetic test signal.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Shared constants for attack types
STANDARD_ATTACK_TYPES = Literal["impulse", "sharp", "medium", "slow"]
WAVEFORM_TYPES = Literal["sine", "triangle", "sawtooth", "square"]
SIGNAL_TYPES = Literal["tempo", "onset", "pitch", "complex"]


class BeatAnnotation(BaseModel):
    """Single beat annotation with timing information."""

    time: float = Field(ge=0, description="Beat time in seconds")
    beat_number: int = Field(default=1, ge=1, description="Beat number within bar")
    bar: int = Field(default=1, ge=1, description="Bar/measure number")


class OnsetAnnotation(BaseModel):
    """Single onset annotation with timing and attack information."""

    time: float = Field(ge=0, description="Onset time in seconds")
    attack_ms: float = Field(
        default=1.0, ge=0, description="Attack time in milliseconds"
    )
    onset_type: STANDARD_ATTACK_TYPES = Field(
        default="sharp", description="Type of onset attack"
    )


class PitchAnnotation(BaseModel):
    """Single pitch annotation with timing and frequency information."""

    start_time: float = Field(ge=0, description="Start time in seconds")
    end_time: float = Field(ge=0, description="End time in seconds")
    midi_note: int = Field(ge=0, le=127, description="MIDI note number (0-127)")
    frequency_hz: float = Field(default=0.0, ge=0, description="Frequency in Hz")
    waveform: WAVEFORM_TYPES = Field(default="sine", description="Waveform type")

    @model_validator(mode="after")
    def compute_frequency(self) -> "PitchAnnotation":
        """Compute frequency from MIDI note if not provided."""
        if self.frequency_hz == 0.0:
            self.frequency_hz = 440.0 * (2.0 ** ((self.midi_note - 69) / 12.0))
        return self


class SignalMetadata(BaseModel):
    """Metadata describing a test signal."""

    sample_rate: int = Field(ge=8000, le=192000, description="Sample rate in Hz")
    duration: float = Field(ge=0.1, description="Duration in seconds")
    description: str = Field(description="Human-readable description")
    bpm: float = Field(
        default=0.0, ge=0, le=600, description="Tempo in BPM (0 if not applicable)"
    )
    channels: int = Field(default=1, ge=1, le=2, description="Number of audio channels")


class TestCriteria(BaseModel):
    """Success criteria for test validation."""

    tempo_tolerance_bpm: float = Field(
        default=2.0, ge=0, description="Allowed tempo detection error in BPM"
    )
    beat_timing_tolerance_ms: float = Field(
        default=50.0, ge=0, description="Allowed beat timing error in ms"
    )
    min_detection_rate: float = Field(
        default=0.95, ge=0, le=1, description="Minimum required detection rate"
    )
    onset_timing_tolerance_ms: float = Field(
        default=50.0, ge=0, description="Allowed onset timing error in ms"
    )
    pitch_tolerance_cents: float = Field(
        default=50.0, ge=0, description="Allowed pitch error in cents"
    )


class GroundTruth(BaseModel):
    """Complete ground truth for a test signal."""

    beats: list[BeatAnnotation] = Field(
        default_factory=list, description="Expected beat annotations"
    )
    onsets: list[OnsetAnnotation] = Field(
        default_factory=list, description="Expected onset annotations"
    )
    pitches: list[PitchAnnotation] = Field(
        default_factory=list, description="Expected pitch annotations"
    )


class SignalDefinition(BaseModel):
    """Complete definition of a test signal with ground truth."""

    signal_type: SIGNAL_TYPES = Field(description="Type of signal")
    metadata: SignalMetadata = Field(description="Signal metadata")
    ground_truth: GroundTruth = Field(
        default_factory=GroundTruth, description="Ground truth data"
    )
    test_criteria: TestCriteria = Field(
        default_factory=TestCriteria, description="Test success criteria"
    )

    model_config = {"extra": "forbid"}
