"""
Aubio BeatCheck - Main Textual Application

A comprehensive TUI for testing and validating aubio's audio analysis functions.
"""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import Footer, Header

from .screens.suite_selector import SuiteSelectorScreen
from .screens.analysis import AnalysisScreen
from .screens.results import ResultsScreen


class AubioBeatCheckApp(App):
    """
    Main Aubio BeatCheck application.

    Provides a terminal-based interface for:
    - Selecting test suites or custom audio files
    - Running aubio analysis with real-time progress
    - Viewing detailed results with visualizations
    - Exporting reports
    """

    CSS = """
    Screen {
        background: $surface;
    }
    
    Header {
        background: $primary;
    }
    
    Footer {
        background: $panel;
    }
    
    .title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin: 1;
    }
    
    .card {
        border: solid $primary;
        padding: 1 2;
        margin: 1;
        background: $panel;
    }
    
    .success {
        color: $success;
    }
    
    .warning {
        color: $warning;
    }
    
    .error {
        color: $error;
    }
    """

    TITLE = "Aubio BeatCheck - Audio Analysis Validation"
    SUB_TITLE = "Validate aubio tempo, onset, pitch, and pvoc analysis"

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("s", "select_suite", "Select Suite"),
        Binding("r", "view_results", "View Results"),
        Binding("h", "help", "Help"),
    ]

    SCREENS = {
        "suite_selector": SuiteSelectorScreen,
        "analysis": AnalysisScreen,
        "results": ResultsScreen,
    }

    def __init__(self):
        super().__init__()
        self.current_results = None
        self.current_suite_name = None

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.push_screen("suite_selector")

    def compose(self) -> ComposeResult:
        """Compose the main UI layout."""
        yield Header()
        yield Container()
        yield Footer()

    def action_select_suite(self) -> None:
        """Navigate to suite selection screen."""
        self.push_screen("suite_selector")

    def action_view_results(self) -> None:
        """Navigate to results viewing screen."""
        if self.current_results:
            self.push_screen("results")
        else:
            self.notify("No results available. Run analysis first.", severity="warning")

    def action_help(self) -> None:
        """Show help information."""
        help_text = """
        Aubio BeatCheck Help
        ====================
        
        This application validates aubio's audio analysis functions:
        - Tempo: Beat tracking and BPM estimation
        - Onset: Transient detection
        - Pitch: Fundamental frequency detection
        - Pvoc: Phase vocoder analysis
        
        Navigation:
        - s: Select test suite
        - r: View results
        - q: Quit application
        
        Workflow:
        1. Select a standard test suite or custom audio
        2. Configure analysis parameters
        3. Run analysis and watch real-time progress
        4. Review detailed results with visualizations
        5. Export reports (JSON, markdown, images)
        
        Test Suites:
        - Tempo: BPM range (60-180), timing jitter, click durations
        - Onset: Various attack types, intervals, waveforms
        - Pitch: Chromatic scales, musical intervals, different timbres
        - Rhythmic: Syncopation, polyrhythms, pattern densities
        - Complex: Combined signals with noise at various SNR levels
        
        For more information, visit the documentation.
        """
        self.push_screen("help", help_text)

    def start_analysis(self, suite_name: str, config: dict) -> None:
        """
        Start analysis with selected suite and configuration.

        Args:
            suite_name: Name of test suite to run
            config: Analysis configuration dictionary
        """
        self.current_suite_name = suite_name
        from aubio_beatcheck.ui.screens.analysis import AnalysisScreen
        self.push_screen(AnalysisScreen(suite_name, config))

    def store_results(self, results: dict) -> None:
        """
        Store analysis results for viewing.

        Args:
            results: Analysis results dictionary
        """
        self.current_results = results
        self.notify("Analysis complete! Press 'r' to view results.", severity="information")


def main():
    """Main entry point for the application."""
    app = AubioBeatCheckApp()
    app.run()


if __name__ == "__main__":
    main()
