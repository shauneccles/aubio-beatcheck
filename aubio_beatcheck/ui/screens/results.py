"""Results viewing screen with detailed metrics and visualizations."""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Label, Static, TabbedContent, TabPane


class ResultsScreen(Screen):
    """Screen for viewing analysis results."""

    CSS = """
    ResultsScreen {
        align: center middle;
    }
    
    #results-container {
        width: 95;
        height: auto;
        border: solid $primary;
        padding: 2;
        background: $panel;
    }
    
    .results-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    .summary-card {
        border: solid $accent;
        padding: 1;
        margin: 1;
        background: $surface;
    }
    
    .metric {
        margin: 0 2;
    }
    
    .metric-label {
        color: $text-muted;
    }
    
    .metric-value {
        color: $text;
        text-style: bold;
    }
    
    DataTable {
        height: 25;
        margin: 1 0;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the results viewing UI."""
        with Container(id="results-container"):
            yield Label("Analysis Results", classes="results-title")

            # Get results from app
            results_data = self.app.current_results

            if not results_data:
                yield Static("No results available.", classes="results-title")
                yield Button("Back", variant="default", id="back")
                return

            suite_name = results_data.get("suite", "Unknown")
            results = results_data.get("results", [])

            yield Static(f"Suite: {suite_name.title()}", classes="results-title")

            with TabbedContent():
                with TabPane("Summary"):
                    yield self._compose_summary(results)

                with TabPane("Detailed Results"):
                    yield self._compose_detailed_results(results)

                with TabPane("Performance"):
                    yield self._compose_performance(results)

            with Horizontal():
                yield Button("Export Results", variant="primary", id="export")
                yield Button("Run New Analysis", variant="default", id="new-analysis")
                yield Button("Back", variant="default", id="back")

    def _compose_summary(self, results: list) -> ComposeResult:
        """Compose summary statistics."""
        with Vertical():
            # Overall statistics
            total_signals = len(results)
            successful = sum(1 for r in results if r.get("success", False))
            failed = total_signals - successful

            with Container(classes="summary-card"):
                yield Label("Overall Statistics")
                yield Static(f"Total Signals: {total_signals}", classes="metric")
                yield Static(
                    f"[green]Successful:[/green] {successful}",
                    classes="metric",
                    markup=True,
                )
                yield Static(f"[red]Failed:[/red] {failed}", classes="metric", markup=True)

            # Category breakdown
            categories = {}
            for r in results:
                cat = r.get("category", "unknown")
                if cat not in categories:
                    categories[cat] = {"total": 0, "success": 0}
                categories[cat]["total"] += 1
                if r.get("success"):
                    categories[cat]["success"] += 1

            with Container(classes="summary-card"):
                yield Label("By Category")
                for cat, stats in categories.items():
                    yield Static(
                        f"{cat.title()}: {stats['success']}/{stats['total']} successful",
                        classes="metric",
                    )

    def _compose_detailed_results(self, results: list) -> ComposeResult:
        """Compose detailed results table."""
        with Vertical():
            table = DataTable()
            table.add_columns("Signal", "Category", "Tempo (BPM)", "Onsets", "Pitches", "Status")

            for result in results:
                table.add_row(
                    result.get("signal_name", "Unknown"),
                    result.get("category", "?"),
                    f"{result.get('tempo_bpm', 0):.1f}" if "tempo_bpm" in result else "-",
                    str(result.get("onset_count", "-")),
                    str(result.get("pitch_count", "-")),
                    "✓" if result.get("success") else "✗",
                )

            yield table

    def _compose_performance(self, results: list) -> ComposeResult:
        """Compose performance metrics."""
        with Vertical():
            # Calculate aggregate performance stats
            tempo_times = []
            onset_times = []
            pitch_times = []

            for r in results:
                if "tempo_stats" in r:
                    tempo_times.append(r["tempo_stats"]["mean_frame_time_us"])
                if "onset_stats" in r:
                    onset_times.append(r["onset_stats"]["mean_frame_time_us"])
                if "pitch_stats" in r:
                    pitch_times.append(r["pitch_stats"]["mean_frame_time_us"])

            with Container(classes="summary-card"):
                yield Label("Average Frame Processing Time (microseconds)")

                if tempo_times:
                    avg_tempo = sum(tempo_times) / len(tempo_times)
                    yield Static(f"Tempo: {avg_tempo:.1f} μs", classes="metric")

                if onset_times:
                    avg_onset = sum(onset_times) / len(onset_times)
                    yield Static(f"Onset: {avg_onset:.1f} μs", classes="metric")

                if pitch_times:
                    avg_pitch = sum(pitch_times) / len(pitch_times)
                    yield Static(f"Pitch: {avg_pitch:.1f} μs", classes="metric")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "new-analysis":
            self.app.action_select_suite()
        elif event.button.id == "export":
            self.notify("Export functionality not yet implemented.", severity="info")
