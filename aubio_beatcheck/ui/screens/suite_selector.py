"""Suite selection screen for choosing test suites."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Label, Select, Static, TabbedContent, TabPane


class SuiteSelectorScreen(Screen):
    """Screen for selecting standard test suites or custom audio."""

    CSS = """
    SuiteSelectorScreen {
        align: center middle;
    }
    
    #suite-container {
        width: 80;
        max-height: 90%;
        border: solid $primary;
        padding: 1;
        background: $panel;
    }
    
    .suite-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    .suite-description {
        margin: 1 0;
        color: $text-muted;
    }
    
    .option-row {
        height: auto;
        margin: 1 0;
    }
    
    Button {
        margin: 1;
    }
    
    Select {
        width: 100%;
        margin: 1 0;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the suite selector UI."""
        with Container(id="suite-container"):
            yield Label("Select Test Suite", classes="suite-title")
            yield Static(
                "Choose a standard test suite or load custom audio files",
                classes="suite-description",
            )

            with TabbedContent():
                with TabPane("Standard Suites"):
                    yield from self._compose_standard_suites()

                with TabPane("Custom Audio"):
                    yield from self._compose_custom_audio()

    def _compose_standard_suites(self) -> ComposeResult:
        """Compose standard suite selection UI."""
        yield Label("Standard Test Suites", classes="suite-title")

        # Suite selection dropdown
        suite_options = [
            ("All Suites", "all"),
            ("Tempo/Beat Tracking", "tempo"),
            ("Onset Detection", "onset"),
            ("Pitch Detection", "pitch"),
            ("Rhythmic Patterns", "rhythmic"),
            ("Complex Signals", "complex"),
        ]
        yield Select(
            options=[(label, value) for label, value in suite_options],
            prompt="Select a test suite...",
            id="suite-select",
        )

        # Suite descriptions
        yield Static(
            "📊 Tempo/Beat Tracking: BPM detection (60-180 BPM)\n"
            "🎯 Onset Detection: Transient detection with various attacks\n"
            "🎵 Pitch Detection: Pitch tracking across musical range\n"
            "🥁 Rhythmic Patterns: Complex patterns & polyrhythms\n"
            "🎛️  Complex Signals: Beats+melody+noise at various SNR",
            classes="suite-description",
        )

        # Configuration options
        with Horizontal(classes="option-row"):
            yield Label("Signal Duration (seconds): ")
            yield Select(
                options=[
                    ("5 seconds", "5"),
                    ("10 seconds (default)", "10"),
                    ("15 seconds", "15"),
                    ("30 seconds", "30"),
                ],
                value="10",
                id="duration-select",
            )

        # Action buttons
        with Horizontal():
            yield Button("Run Analysis", variant="primary", id="run-standard")
            yield Button("Cancel", variant="default", id="cancel")

    def _compose_custom_audio(self) -> ComposeResult:
        """Compose custom audio loading UI."""
        yield Label("Load Custom Audio", classes="suite-title")

        yield Static(
            "Load your own audio files for aubio analysis.\n\n"
            "Supported formats: WAV, FLAC, OGG, MP3\n\n"
            "Note: Custom audio will be analyzed without ground truth,\n"
            "so only detection metrics (not accuracy) will be available.\n\n"
            "Optional: Provide a JSON file with ground truth annotations.",
            classes="suite-description",
        )

        # File selection
        yield Button("Browse for Audio File...", id="browse-audio")
        yield Static("No file selected", id="selected-file")

        yield Button("Browse for Ground Truth JSON (optional)...", id="browse-json")
        yield Static("No ground truth file", id="selected-json")

        # Action buttons
        with Horizontal():
            yield Button("Analyze File", variant="primary", id="run-custom", disabled=True)
            yield Button("Cancel", variant="default", id="cancel")

    @on(Button.Pressed, "#run-standard")
    async def run_standard_suite(self, event: Button.Pressed) -> None:
        """Handle standard suite analysis."""
        suite_select = self.query_one("#suite-select", Select)
        duration_select = self.query_one("#duration-select", Select)

        if suite_select.value is Select.BLANK or suite_select.value is None:
            self.notify("Please select a test suite", severity="warning")
            return

        config = {
            "duration": float(duration_select.value or "10"),
            "type": "standard",
        }

        # Start analysis
        self.app.start_analysis(str(suite_select.value), config)

    @on(Button.Pressed, "#browse-audio")
    async def browse_audio_file(self, event: Button.Pressed) -> None:
        """Open file browser for audio selection."""
        # Note: Textual doesn't have built-in file browser
        # This would require platform-specific implementation or third-party library
        self.notify("File browser not yet implemented. Use CLI path input.", severity="info")

    @on(Button.Pressed, "#browse-json")
    async def browse_json_file(self, event: Button.Pressed) -> None:
        """Open file browser for ground truth JSON selection."""
        self.notify("File browser not yet implemented. Use CLI path input.", severity="info")

    @on(Button.Pressed, "#run-custom")
    async def run_custom_analysis(self, event: Button.Pressed) -> None:
        """Handle custom audio analysis."""
        self.notify("Custom audio analysis not yet implemented.", severity="info")

    @on(Button.Pressed, "#cancel")
    async def cancel_selection(self, event: Button.Pressed) -> None:
        """Cancel and return to previous screen."""
        self.app.pop_screen()
