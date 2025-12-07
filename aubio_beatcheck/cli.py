#!/usr/bin/env python3
"""Aubio BeatCheck - CLI Entry Point.

Command-line interface for the aubio validation application.
Supports both web UI mode and CLI mode for AI agent integration.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from aubio_beatcheck.core.analyzers import (
    AnalyzerConfig,
    OnsetAnalyzer,
    PitchAnalyzer,
    TempoAnalyzer,
)
from aubio_beatcheck.core.evaluation import EvaluationMetrics, Evaluator
from aubio_beatcheck.suites.standard import StandardSuites

# --- Pydantic Models for CLI Output ---


class TestInputConfig(BaseModel):
    """Configuration used to run the test suite."""

    suite: str = Field(description="Test suite name")
    duration: float = Field(description="Signal duration in seconds")
    sample_rate: int = Field(default=44100, description="Sample rate in Hz")


class SignalResult(BaseModel):
    """Analysis result for a single signal."""

    signal_name: str
    category: str
    status: str
    tempo_bpm: float | None = None
    beat_count: int | None = None
    onset_count: int | None = None
    pitch_count: int | None = None
    detected_events: list[float] = Field(default_factory=list)
    ground_truth_events: list[float] = Field(default_factory=list)
    evaluation: EvaluationMetrics | None = None
    performance_mean_us: float | None = None
    performance_p95_us: float | None = None
    error: str | None = None
    plot_path: str | None = None


class AnalysisResultsOutput(BaseModel):
    """Complete analysis results for AI consumption."""

    suite: str
    total_signals: int
    completed: int
    failed: int
    signals: list[SignalResult]


def setup_logging(verbose: bool = False):
    """Configure logging."""
    logger.remove()

    log_file = Path.cwd() / "logs" / "aubio-beatcheck.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        rotation="10 MB",
        retention="7 days",
        level="DEBUG" if verbose else "INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
    )

    if verbose:
        logger.add(sys.stderr, level="DEBUG")


def run_analysis(suite_id: str, duration: float, output_dir: Path) -> int:
    """Run analysis and output artifacts."""
    logger.info(f"Running suite '{suite_id}' with duration={duration}s")

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Get signals
    signals = StandardSuites.get_suite(suite_id, duration=duration)
    logger.info(f"Loaded {len(signals)} test signals")

    # Save test input config
    test_input = TestInputConfig(
        suite=suite_id,
        duration=duration,
        sample_rate=44100,
    )
    (output_dir / "test_input.json").write_text(test_input.model_dump_json(indent=2))

    # Prepare ground truth output
    ground_truth_data = {}
    results: list[SignalResult] = []

    for signal in signals:
        logger.info(f"Analyzing: {signal.name}")

        # Add to ground truth
        ground_truth_data[signal.name] = {
            "category": signal.category,
            "signal_definition": signal.signal_def.model_dump(),
        }

        # Run analysis
        result = analyze_signal(signal, plots_dir)
        results.append(result)

    # Save ground truth
    (output_dir / "ground_truth.json").write_text(
        json.dumps(ground_truth_data, indent=2, default=str)
    )

    # Save analysis results
    completed = sum(1 for r in results if r.status == "completed")
    failed = len(results) - completed

    analysis_output = AnalysisResultsOutput(
        suite=suite_id,
        total_signals=len(results),
        completed=completed,
        failed=failed,
        signals=results,
    )
    (output_dir / "analysis_results.json").write_text(
        analysis_output.model_dump_json(indent=2)
    )

    # Save evaluation summary
    evaluation_summary = {
        "suite": suite_id,
        "summary": {
            "total": len(results),
            "completed": completed,
            "failed": failed,
            "pass_rate": completed / len(results) if results else 0,
        },
        "per_signal": {
            r.signal_name: r.evaluation.model_dump() if r.evaluation else None
            for r in results
        },
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(evaluation_summary, indent=2, default=str)
    )

    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Completed: {completed}/{len(results)}")

    return 0 if failed == 0 else 1


def analyze_signal(signal, plots_dir: Path) -> SignalResult:
    """Analyze a single signal and return results."""
    try:
        config = AnalyzerConfig(sample_rate=int(signal.signal_def.metadata.sample_rate))
        audio = np.asarray(signal.audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.flatten()

        result = SignalResult(
            signal_name=signal.name,
            category=signal.category,
            status="completed",
        )

        if signal.category in ("tempo", "complex"):
            analyzer = TempoAnalyzer(config)
            beats, bpm = analyzer.analyze(audio)
            gt_beats = [b.time for b in signal.signal_def.ground_truth.beats]

            eval_metrics = Evaluator.evaluate_events(
                beats,
                gt_beats,
                tolerance_ms=signal.signal_def.test_criteria.beat_timing_tolerance_ms,
            )

            result.tempo_bpm = bpm
            result.beat_count = len(beats)
            result.detected_events = beats
            result.ground_truth_events = gt_beats
            result.evaluation = eval_metrics
            result.performance_mean_us = analyzer.stats.mean_us
            result.performance_p95_us = analyzer.stats.p95_us

        elif signal.category == "onset":
            analyzer = OnsetAnalyzer(config)
            onsets = analyzer.analyze(audio)
            gt_onsets = [o.time for o in signal.signal_def.ground_truth.onsets]

            eval_metrics = Evaluator.evaluate_events(
                onsets,
                gt_onsets,
                tolerance_ms=signal.signal_def.test_criteria.onset_timing_tolerance_ms,
            )

            result.onset_count = len(onsets)
            result.detected_events = onsets
            result.ground_truth_events = gt_onsets
            result.evaluation = eval_metrics
            result.performance_mean_us = analyzer.stats.mean_us
            result.performance_p95_us = analyzer.stats.p95_us

        elif signal.category == "pitch":
            analyzer = PitchAnalyzer(config)
            pitches = analyzer.analyze(audio)
            result.pitch_count = len(pitches)
            result.detected_events = [p[0] for p in pitches]
            result.performance_mean_us = analyzer.stats.mean_us
            result.performance_p95_us = analyzer.stats.p95_us

        # Generate plot
        # Import here to avoid circular dependency with web_api
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))

        plot_path = plots_dir / f"{signal.name}.png"

        if signal.category == "pitch":
            # Use pitch-specific piano roll visualization
            from web_api.plotting import generate_pitch_analysis_plot

            # Get ground truth pitch annotations as dicts
            gt_pitches = [
                {
                    "start_time": p.start_time,
                    "end_time": p.end_time,
                    "midi_note": p.midi_note,
                    "frequency_hz": p.frequency_hz,
                }
                for p in signal.signal_def.ground_truth.pitches
            ]

            # Get detected pitches (already have full data from analyzer)
            detected_pitch_data = pitches if signal.category == "pitch" else []

            # Get tolerance from test criteria
            tolerance = signal.signal_def.test_criteria.pitch_tolerance_cents or 50.0

            plot_bytes = generate_pitch_analysis_plot(
                signal_name=signal.name,
                audio=signal.audio,
                sample_rate=int(signal.signal_def.metadata.sample_rate),
                ground_truth_pitches=gt_pitches,
                detected_pitches=detected_pitch_data,
                pitch_tolerance_cents=tolerance,
            )
        else:
            # Use standard event-based visualization for tempo/onset
            from web_api.plotting import generate_analysis_plot

            event_type = "Beats" if signal.category == "tempo" else "Onsets"

            plot_bytes = generate_analysis_plot(
                signal_name=signal.name,
                audio=signal.audio,
                sample_rate=int(signal.signal_def.metadata.sample_rate),
                ground_truth_events=result.ground_truth_events,
                detected_events=result.detected_events,
                event_type=event_type,
            )

        plot_path.write_bytes(plot_bytes)
        result.plot_path = str(plot_path)

        return result

    except Exception as e:
        logger.error(f"Failed to analyze {signal.name}: {e}")
        return SignalResult(
            signal_name=signal.name,
            category=signal.category,
            status="failed",
            error=str(e),
        )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Aubio BeatCheck - Validate aubio audio analysis functions",
        epilog="For AI agent integration, use --suite and --output flags.",
    )

    parser.add_argument("--version", action="version", version="aubio-beatcheck 0.1.0")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run analysis suite")
    run_parser.add_argument(
        "--suite",
        "-s",
        choices=["tempo", "onset", "pitch", "rhythmic", "complex", "all"],
        default="all",
        help="Test suite to run (default: all)",
    )
    run_parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=10.0,
        help="Signal duration in seconds (default: 10.0)",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("./results"),
        help="Output directory for artifacts (default: ./results)",
    )

    # Web command
    web_parser = subparsers.add_parser("web", help="Start web UI server")
    web_parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    web_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )

    # Save instructions command
    subparsers.add_parser(
        "save-instructions",
        help="Save AI agent instructions to current directory",
    )

    args = parser.parse_args()
    setup_logging(verbose=getattr(args, "verbose", False))

    if args.command == "run":
        return run_analysis(args.suite, args.duration, args.output)
    elif args.command == "web":
        import uvicorn

        uvicorn.run("web_api.main:app", host=args.host, port=args.port, reload=True)
        return 0
    elif args.command == "save-instructions":
        return save_agent_instructions()
    else:
        parser.print_help()
        return 0


def save_agent_instructions() -> int:
    """Save agent instructions to current directory."""
    import importlib.resources

    try:
        # Read from package resources
        files = importlib.resources.files("aubio_beatcheck")
        content = (files / "AGENT_INSTRUCTIONS.md").read_text(encoding="utf-8")

        # Write to current directory
        output_path = Path.cwd() / "AGENT_INSTRUCTIONS.md"
        output_path.write_text(content, encoding="utf-8")

        print(f"Agent instructions saved to: {output_path}")
        return 0

    except Exception as e:
        print(f"Error saving instructions: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
