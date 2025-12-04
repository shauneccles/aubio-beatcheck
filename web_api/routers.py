import numpy as np
from typing import List, Dict
from litestar import Controller, get, post, Response
from litestar.background_tasks import BackgroundTask
from litestar.exceptions import NotFoundException
from loguru import logger

from aubio_beatcheck.suites.standard import StandardSuites
from aubio_beatcheck.core.analyzers import (
    AnalyzerConfig,
    TempoAnalyzer,
    OnsetAnalyzer,
    PitchAnalyzer,
)
from .models import AnalysisRequest, AnalysisResult, SuiteInfo

from aubio_beatcheck.core.evaluation import Evaluator

# In-memory storage for results (for simplicity)
results_store: Dict[str, List[AnalysisResult]] = {}


async def process_analysis(suite_id: str, request: AnalysisRequest) -> None:
    """Process analysis in background."""
    logger.info(f"Starting analysis for {suite_id}")
    try:
        signals = StandardSuites.get_suite(suite_id, duration=request.duration)
        results = []

        for signal in signals:
            result_data = {
                "signal_name": signal.name,
                "category": signal.category,
                "success": True,
                "metrics": {},
            }

            try:
                # Configure analyzer
                config = AnalyzerConfig(
                    sample_rate=int(signal.signal_def.metadata.sample_rate)
                )

                # Prepare audio
                audio = np.asarray(signal.audio, dtype=np.float32)
                if audio.ndim > 1:
                    audio = audio.flatten()

                # Run analysis based on category
                if signal.category in ("tempo", "complex"):
                    analyzer = TempoAnalyzer(config)
                    beats, bpm = analyzer.analyze(audio)

                    # Get ground truth beats
                    gt_beats = [b.time for b in signal.signal_def.ground_truth.beats]

                    # Evaluate
                    eval_metrics = Evaluator.evaluate_events(
                        beats,
                        gt_beats,
                        tolerance_ms=signal.signal_def.test_criteria.beat_timing_tolerance_ms,
                    )

                    result_data["metrics"]["tempo_bpm"] = bpm
                    result_data["metrics"]["beat_count"] = len(beats)
                    result_data["metrics"]["stats"] = {
                        "mean_us": analyzer.stats.mean_us,
                        "p95_us": analyzer.stats.p95_us,
                    }
                    result_data["metrics"]["evaluation"] = {
                        "precision": eval_metrics.precision,
                        "recall": eval_metrics.recall,
                        "f_measure": eval_metrics.f_measure,
                        "mae_ms": eval_metrics.mean_absolute_error_ms,
                        "false_positives": eval_metrics.false_positives,
                        "false_negatives": eval_metrics.false_negatives,
                    }
                    # Store raw events for plotting
                    result_data["metrics"]["detected_events"] = beats
                    result_data["metrics"]["ground_truth_events"] = gt_beats

                elif signal.category == "onset":
                    analyzer = OnsetAnalyzer(config)
                    onsets = analyzer.analyze(audio)

                    # Get ground truth onsets
                    gt_onsets = [o.time for o in signal.signal_def.ground_truth.onsets]

                    # Evaluate
                    eval_metrics = Evaluator.evaluate_events(
                        onsets,
                        gt_onsets,
                        tolerance_ms=signal.signal_def.test_criteria.onset_timing_tolerance_ms,
                    )

                    result_data["metrics"]["onset_count"] = len(onsets)
                    result_data["metrics"]["stats"] = {
                        "mean_us": analyzer.stats.mean_us,
                        "p95_us": analyzer.stats.p95_us,
                    }
                    result_data["metrics"]["evaluation"] = {
                        "precision": eval_metrics.precision,
                        "recall": eval_metrics.recall,
                        "f_measure": eval_metrics.f_measure,
                        "mae_ms": eval_metrics.mean_absolute_error_ms,
                        "false_positives": eval_metrics.false_positives,
                        "false_negatives": eval_metrics.false_negatives,
                    }
                    # Store raw events for plotting
                    result_data["metrics"]["detected_events"] = onsets
                    result_data["metrics"]["ground_truth_events"] = gt_onsets

                elif signal.category == "pitch":
                    analyzer = PitchAnalyzer(config)
                    pitches = analyzer.analyze(audio)
                    result_data["metrics"]["pitch_count"] = len(pitches)
                    result_data["metrics"]["stats"] = {
                        "mean_us": analyzer.stats.mean_us,
                        "p95_us": analyzer.stats.p95_us,
                    }
                    # For pitch, we have (time, note, confidence) tuples
                    # For simple plotting, we might just plot the times?
                    # Or maybe skip plotting for pitch for now as it's 2D (time vs pitch)
                    # Let's just store the times for consistency if we want to plot "events"
                    result_data["metrics"]["detected_events"] = [p[0] for p in pitches]
                    # GT pitches is list of PitchAnnotation (start, end, note)
                    # This is harder to plot as simple vertical lines.
                    # We'll skip GT events for pitch plotting for now or implement a better pitch plotter later.

                results.append(
                    AnalysisResult(
                        signal_name=signal.name,
                        status="completed",
                        metrics=result_data["metrics"],
                    )
                )
                logger.info(f"Analyzed {signal.name}")

            except Exception as e:
                logger.error(f"Failed to analyze {signal.name}: {e}")
                results.append(
                    AnalysisResult(
                        signal_name=signal.name, status="failed", error=str(e)
                    )
                )

        results_store[suite_id] = results
        logger.info(f"Analysis for {suite_id} completed")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")


class AnalysisController(Controller):
    path = "/api"

    @get("/suites")
    async def list_suites(self) -> List[SuiteInfo]:
        """List available test suites."""
        return [
            SuiteInfo(
                id="tempo",
                name="Tempo/Beat Tracking",
                description="Tests BPM detection and beat tracking",
            ),
            SuiteInfo(
                id="onset",
                name="Onset Detection",
                description="Tests transient detection",
            ),
            SuiteInfo(
                id="pitch", name="Pitch Detection", description="Tests pitch tracking"
            ),
            SuiteInfo(
                id="rhythmic",
                name="Rhythmic Patterns",
                description="Tests complex rhythms",
            ),
            SuiteInfo(
                id="complex",
                name="Complex Signals",
                description="Tests combined analysis",
            ),
            SuiteInfo(id="all", name="All Suites", description="Run all test suites"),
        ]

    @post("/analyze/{suite_id:str}")
    async def run_analysis(
        self, suite_id: str, data: AnalysisRequest
    ) -> Response[Dict[str, str]]:
        """Start analysis for a specific suite."""
        if suite_id not in ["tempo", "onset", "pitch", "rhythmic", "complex", "all"]:
            raise NotFoundException(detail="Suite not found")

        # Start analysis in background
        return Response(
            content={"status": "started", "suite_id": suite_id},
            background=BackgroundTask(process_analysis, suite_id, data),
        )

    @get("/results/{suite_id:str}")
    async def get_results(self, suite_id: str) -> List[AnalysisResult]:
        """Get results for a suite."""
        return results_store.get(suite_id, [])

    @get("/results/{suite_id:str}/{signal_name:str}/plot", media_type="image/png")
    async def get_result_plot(self, suite_id: str, signal_name: str) -> Response[bytes]:
        """Get waveform plot for a specific result."""
        # 1. Get the signal definition (re-generate)
        try:
            signals = StandardSuites.get_suite(suite_id)
            signal = next((s for s in signals if s.name == signal_name), None)
            if not signal:
                raise NotFoundException(detail="Signal not found")
        except ValueError:
            raise NotFoundException(detail="Suite not found")

        # 2. Get the analysis result
        results = results_store.get(suite_id, [])
        result = next((r for r in results if r.signal_name == signal_name), None)

        if not result or not result.metrics:
            # If no result yet, we can still plot the ground truth?
            # Or just 404? Let's 404 for now as we need detected events for the full plot
            raise NotFoundException(detail="Analysis result not found")

        # 3. Generate plot
        from .plotting import generate_analysis_plot

        detected_events = result.metrics.get("detected_events", [])
        ground_truth_events = result.metrics.get("ground_truth_events", [])

        # Determine event type label
        event_type = "Events"
        if signal.category == "tempo":
            event_type = "Beats"
        elif signal.category == "onset":
            event_type = "Onsets"

        plot_bytes = generate_analysis_plot(
            signal_name=signal.name,
            audio=signal.audio,
            sample_rate=int(signal.signal_def.metadata.sample_rate),
            ground_truth_events=ground_truth_events,
            detected_events=detected_events,
            event_type=event_type,
        )

        return Response(plot_bytes, media_type="image/png")
