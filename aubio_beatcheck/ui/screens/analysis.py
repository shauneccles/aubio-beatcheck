"""Analysis progress screen with real-time updates."""

import asyncio
import numpy as np
from loguru import logger

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Button, Label, ProgressBar, Static, DataTable
from textual.worker import Worker, WorkerState

from aubio_beatcheck.core.analyzers import (
    AnalyzerConfig,
    OnsetAnalyzer,
    PitchAnalyzer,
    TempoAnalyzer,
)
from aubio_beatcheck.core.ground_truth import SignalDefinition
from aubio_beatcheck.suites.standard import StandardSuites


class AnalysisScreen(Screen):
    """Screen for running analysis with real-time progress."""

    CSS = """
    AnalysisScreen {
        align: center middle;
    }
    
    #analysis-container {
        width: 90;
        height: auto;
        border: solid $primary;
        padding: 2;
        background: $panel;
    }
    
    .analysis-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    ProgressBar {
        margin: 1 0;
    }
    
    DataTable {
        height: 20;
        margin: 1 0;
    }
    
    .status-message {
        margin: 1 0;
        text-align: center;
    }
    """

    def __init__(self, suite_name: str, config: dict):
        super().__init__()
        self.suite_name = suite_name
        self.config = config
        self.results = []
        self.worker = None

    def compose(self) -> ComposeResult:
        """Compose the analysis progress UI."""
        with Container(id="analysis-container"):
            yield Label(f"Running Analysis: {self.suite_name.title()}", classes="analysis-title")

            yield ProgressBar(id="progress-bar", total=100)
            yield Static("Initializing...", id="status-message", classes="status-message")

            # Results table
            table = DataTable(id="results-table")
            table.add_columns("Signal", "Tempo", "Onset", "Pitch", "Status")
            yield table

            yield Button("Cancel Analysis", variant="error", id="cancel-analysis")

    async def on_mount(self) -> None:
        """Start analysis when screen is mounted."""
        await self.run_analysis()

    async def run_analysis(self) -> None:
        """Run aubio analysis on selected test suite."""
        logger.info(f"Starting analysis for suite: {self.suite_name}")
        try:
            # Get test signals
            status = self.query_one("#status-message", Static)
            status.update("Loading test signals...")

            signals = StandardSuites.get_suite(
                self.suite_name, duration=self.config.get("duration", 10.0)
            )

            if not signals:
                logger.warning(f"No signals found in suite: {self.suite_name}")
                status.update("No signals found in suite!")
                return

            logger.info(f"Loaded {len(signals)} test signals")
            # Initialize progress
            progress = self.query_one("#progress-bar", ProgressBar)
            progress.update(total=len(signals))

            table = self.query_one("#results-table", DataTable)

            # Analyze each signal
            for idx, signal in enumerate(signals):
                logger.debug(f"Analyzing signal {idx + 1}/{len(signals)}: {signal.name}")
                status.update(f"Analyzing {idx + 1}/{len(signals)}: {signal.name}")

                # Run analysis
                result = await self.analyze_signal(signal)
                self.results.append(result)

                # Update table
                table.add_row(
                    signal.name,
                    f"{result.get('tempo_bpm', 0):.1f} BPM",
                    f"{result.get('onset_count', 0)} onsets",
                    f"{result.get('pitch_count', 0)} pitches",
                    "✓" if result.get("success") else "✗",
                )

                # Update progress
                progress.update(progress=idx + 1)

                # Allow UI to update
                await asyncio.sleep(0.1)

            # Complete
            logger.success(f"Analysis complete! Processed {len(signals)} signals")
            status.update(f"Analysis complete! Processed {len(signals)} signals.")
            self.query_one("#cancel-analysis", Button).label = "View Results →"

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            logger.exception("Full traceback:")
            status = self.query_one("#status-message", Static)
            status.update(f"Error during analysis: {str(e)}")
            self.notify(f"Analysis failed: {str(e)}", severity="error")

    async def analyze_signal(self, signal) -> dict:
        """
        Analyze a single signal with aubio.

        Args:
            signal: TestSignal object

        Returns:
            Dictionary with analysis results
        """
        result = {
            "signal_name": signal.name,
            "category": signal.category,
            "success": True,
        }

        try:
            # Configure analyzers
            config = AnalyzerConfig(
                fft_size=2048, hop_size=512, sample_rate=int(signal.signal_def.metadata.sample_rate)
            )
            
            logger.debug(f"Analyzing {signal.name}: category={signal.category}, "
                        f"sample_rate={config.sample_rate}, audio_shape={signal.audio.shape}")
            
            # Ensure audio is the right type and shape
            audio = np.asarray(signal.audio, dtype=np.float32)
            if audio.ndim > 1:
                logger.warning(f"Audio has {audio.ndim} dimensions, flattening")
                audio = audio.flatten()
            
            logger.debug(f"Processed audio: shape={audio.shape}, dtype={audio.dtype}, "
                        f"length={len(audio)/config.sample_rate:.2f}s")

            # Tempo analysis
            if signal.category in ("tempo", "complex"):
                logger.debug(f"Running tempo analysis on {signal.name}")
                tempo_analyzer = TempoAnalyzer(config)
                beats, bpm = tempo_analyzer.analyze(audio)
                result["tempo_bpm"] = bpm
                result["detected_beats"] = len(beats)
                result["tempo_stats"] = {
                    "mean_frame_time_us": tempo_analyzer.stats.mean_us,
                    "p95_frame_time_us": tempo_analyzer.stats.p95_us,
                }
                logger.debug(f"Tempo result: {bpm:.1f} BPM, {len(beats)} beats")

            # Onset analysis
            if signal.category in ("onset", "complex"):
                logger.debug(f"Running onset analysis on {signal.name}")
                onset_analyzer = OnsetAnalyzer(config)
                onsets = onset_analyzer.analyze(audio)
                result["onset_count"] = len(onsets)
                result["onset_stats"] = {
                    "mean_frame_time_us": onset_analyzer.stats.mean_us,
                    "p95_frame_time_us": onset_analyzer.stats.p95_us,
                }
                logger.debug(f"Onset result: {len(onsets)} onsets detected")

            # Pitch analysis
            if signal.category in ("pitch", "complex"):
                logger.debug(f"Running pitch analysis on {signal.name}")
                pitch_analyzer = PitchAnalyzer(config)
                pitches = pitch_analyzer.analyze(audio)
                result["pitch_count"] = len(pitches)
                result["pitch_stats"] = {
                    "mean_frame_time_us": pitch_analyzer.stats.mean_us,
                    "p95_frame_time_us": pitch_analyzer.stats.p95_us,
                }
                logger.debug(f"Pitch result: {len(pitches)} pitch detections")
            
            logger.info(f"Successfully analyzed {signal.name}")

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            logger.error(f"Analysis failed for {signal.name}: {e}")
            logger.error(f"Signal details: shape={signal.audio.shape}, dtype={signal.audio.dtype}")
            logger.exception("Full traceback:")

        return result

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "cancel-analysis":
            if self.results:
                # Analysis complete, show results
                self.app.store_results({"suite": self.suite_name, "results": self.results})
                self.app.pop_screen()
            else:
                # Cancel ongoing analysis
                self.app.pop_screen()
