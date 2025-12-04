"""
Tests for CLI functionality and analysis pipeline.
"""

import json
import pytest
from pathlib import Path
import tempfile
import shutil

from aubio_beatcheck.cli import run_analysis
from aubio_beatcheck.core.evaluation import Evaluator, EvaluationMetrics
from aubio_beatcheck.core.ground_truth import (
    BeatAnnotation,
    OnsetAnnotation,
    SignalMetadata,
    GroundTruth,
    SignalDefinition,
)


class TestPydanticModels:
    """Test Pydantic model validation."""

    def test_beat_annotation_validation(self):
        """Test BeatAnnotation validates correctly."""
        beat = BeatAnnotation(time=1.5, beat_number=2, bar=1)
        assert beat.time == 1.5
        assert beat.beat_number == 2

    def test_beat_annotation_defaults(self):
        """Test BeatAnnotation default values."""
        beat = BeatAnnotation(time=0.0)
        assert beat.beat_number == 1
        assert beat.bar == 1

    def test_onset_annotation_type_validation(self):
        """Test OnsetAnnotation type is validated."""
        onset = OnsetAnnotation(time=0.5, onset_type="sharp")
        assert onset.onset_type == "sharp"

    def test_signal_metadata_validation(self):
        """Test SignalMetadata validates sample rate range."""
        meta = SignalMetadata(
            sample_rate=44100, duration=10.0, description="Test signal"
        )
        assert meta.sample_rate == 44100

    def test_signal_definition_serialization(self):
        """Test SignalDefinition can be serialized to JSON."""
        sig_def = SignalDefinition(
            signal_type="tempo",
            metadata=SignalMetadata(
                sample_rate=44100, duration=5.0, description="Test", bpm=120.0
            ),
            ground_truth=GroundTruth(
                beats=[BeatAnnotation(time=0.5), BeatAnnotation(time=1.0)]
            ),
        )
        json_str = sig_def.model_dump_json()
        assert "tempo" in json_str
        assert "120.0" in json_str


class TestEvaluator:
    """Test Evaluator class."""

    def test_perfect_detection(self):
        """Test perfect detection scenario."""
        gt = [0.5, 1.0, 1.5, 2.0]
        detected = [0.5, 1.0, 1.5, 2.0]

        result = Evaluator.evaluate_events(detected, gt, tolerance_ms=50)

        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f_measure == 1.0
        assert len(result.false_positives) == 0
        assert len(result.false_negatives) == 0

    def test_missed_events(self):
        """Test scenario with missed events."""
        gt = [0.5, 1.0, 1.5, 2.0]
        detected = [0.5, 1.0]  # Missing 1.5 and 2.0

        result = Evaluator.evaluate_events(detected, gt, tolerance_ms=50)

        assert result.recall == 0.5  # 2/4
        assert result.precision == 1.0  # No false positives
        assert len(result.false_negatives) == 2

    def test_false_positives(self):
        """Test scenario with false positives."""
        gt = [0.5, 1.0]
        detected = [0.5, 1.0, 1.5, 2.0]  # Extra at 1.5 and 2.0

        result = Evaluator.evaluate_events(detected, gt, tolerance_ms=50)

        assert result.recall == 1.0
        assert result.precision == 0.5  # 2/4
        assert len(result.false_positives) == 2

    def test_timing_tolerance(self):
        """Test timing tolerance is applied correctly."""
        gt = [1.0]
        detected = [1.04]  # 40ms off

        # Should match with 50ms tolerance
        result1 = Evaluator.evaluate_events(detected, gt, tolerance_ms=50)
        assert result1.recall == 1.0

        # Should not match with 30ms tolerance
        result2 = Evaluator.evaluate_events(detected, gt, tolerance_ms=30)
        assert result2.recall == 0.0

    def test_empty_ground_truth(self):
        """Test with empty ground truth."""
        result = Evaluator.evaluate_events([0.5, 1.0], [], tolerance_ms=50)
        assert result.recall == 1.0
        assert len(result.false_positives) == 2

    def test_empty_detected(self):
        """Test with no detections."""
        result = Evaluator.evaluate_events([], [0.5, 1.0], tolerance_ms=50)
        assert result.precision == 1.0
        assert result.recall == 0.0
        assert len(result.false_negatives) == 2


class TestCLIAnalysis:
    """Integration tests for CLI analysis."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_tempo_suite_outputs(self, temp_output_dir):
        """Test tempo suite generates expected outputs."""
        result = run_analysis("tempo", duration=2.0, output_dir=temp_output_dir)

        # Check all expected files exist
        assert (temp_output_dir / "test_input.json").exists()
        assert (temp_output_dir / "ground_truth.json").exists()
        assert (temp_output_dir / "analysis_results.json").exists()
        assert (temp_output_dir / "evaluation.json").exists()
        assert (temp_output_dir / "plots").is_dir()

        # Verify test_input.json content
        test_input = json.loads((temp_output_dir / "test_input.json").read_text())
        assert test_input["suite"] == "tempo"
        assert test_input["duration"] == 2.0

    def test_onset_suite_outputs(self, temp_output_dir):
        """Test onset suite generates expected outputs."""
        result = run_analysis("onset", duration=2.0, output_dir=temp_output_dir)

        analysis = json.loads((temp_output_dir / "analysis_results.json").read_text())
        assert analysis["suite"] == "onset"
        assert analysis["total_signals"] > 0

    def test_pitch_suite_outputs(self, temp_output_dir):
        """Test pitch suite generates expected outputs."""
        result = run_analysis("pitch", duration=2.0, output_dir=temp_output_dir)

        analysis = json.loads((temp_output_dir / "analysis_results.json").read_text())
        assert analysis["suite"] == "pitch"

    def test_plots_generated(self, temp_output_dir):
        """Test PNG plots are generated for signals."""
        run_analysis("tempo", duration=2.0, output_dir=temp_output_dir)

        plots_dir = temp_output_dir / "plots"
        png_files = list(plots_dir.glob("*.png"))
        assert len(png_files) > 0


class TestEvaluationMetricsSerialization:
    """Test EvaluationMetrics JSON serialization."""

    def test_model_dump(self):
        """Test EvaluationMetrics can be serialized."""
        metrics = EvaluationMetrics(
            precision=0.9,
            recall=0.85,
            f_measure=0.87,
            mean_absolute_error_ms=12.5,
            false_positives=[1.5, 2.5],
            false_negatives=[3.0],
        )

        json_str = metrics.model_dump_json()
        data = json.loads(json_str)

        assert data["precision"] == 0.9
        assert data["recall"] == 0.85
        assert len(data["false_positives"]) == 2
