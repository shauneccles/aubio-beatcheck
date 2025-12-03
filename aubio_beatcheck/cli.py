#!/usr/bin/env python3
"""
Aubio BeatCheck - CLI Entry Point

Command-line interface for the aubio validation application.
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

from aubio_beatcheck.ui.app import AubioBeatCheckApp


def setup_logging():
    """Configure loguru to write to a file since Textual hides console output."""
    # Remove default handler
    logger.remove()
    
    # Add file handler with rotation
    log_file = Path.home() / ".aubio-beatcheck" / "logs" / "aubio-beatcheck.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file,
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    )
    
    logger.info("=== Aubio BeatCheck Started ===")
    logger.info(f"Log file: {log_file}")


def main():
    """Main CLI entry point."""
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Aubio BeatCheck - Validate aubio audio analysis functions",
        epilog="For more information, see the documentation at https://github.com/...",
    )

    parser.add_argument(
        "--version", action="version", version="aubio-beatcheck 0.1.0"
    )

    parser.add_argument(
        "--suite",
        "-s",
        choices=["tempo", "onset", "pitch", "rhythmic", "complex", "all"],
        help="Run specific test suite from CLI (non-interactive)",
    )

    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=10.0,
        help="Signal duration in seconds (default: 10.0)",
    )

    parser.add_argument(
        "--export",
        "-e",
        type=Path,
        help="Export results to directory (requires --suite)",
    )

    parser.add_argument(
        "--audio",
        "-a",
        type=Path,
        help="Analyze custom audio file",
    )

    parser.add_argument(
        "--ground-truth",
        "-g",
        type=Path,
        help="Ground truth JSON file for custom audio",
    )

    args = parser.parse_args()

    # Non-interactive mode
    if args.suite:
        logger.info(f"Running in CLI mode with suite: {args.suite}")
        return run_cli_mode(args)

    # Interactive TUI mode
    logger.info("Starting interactive TUI mode")
    app = AubioBeatCheckApp()
    try:
        app.run()
        return 0
    except Exception as e:
        logger.exception(f"Application crashed: {e}")
        raise


def run_cli_mode(args):
    """Run in non-interactive CLI mode."""
    print(f"Running suite: {args.suite}")
    print(f"Signal duration: {args.duration}s")

    try:
        from aubio_beatcheck.suites.standard import StandardSuites

        # Get signals
        signals = StandardSuites.get_suite(args.suite, duration=args.duration)
        print(f"Loaded {len(signals)} test signals")

        # TODO: Run analysis and export results
        print("\nCLI mode analysis not yet fully implemented.")
        print("Use the interactive TUI mode (run without --suite) for now.")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
