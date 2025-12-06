"""Continuous Feedback Loop for Performance Monitoring.

Provides performance monitoring, drift detection, automatic rollback,
and trend analysis for closed-loop optimization of aubio parameters.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from aubio_beatcheck.core.analyzers import AnalyzerConfig
from aubio_beatcheck.core.benchmark_db import BenchmarkDB, BenchmarkRun


class ActionType(str, Enum):
    """Types of feedback actions triggered by performance changes."""

    MAINTAIN = "maintain"  # No action needed, performance stable
    ROLLBACK = "rollback"  # Performance dropped, revert to previous config
    UPDATE_BASELINE = "update_baseline"  # Performance improved, save new baseline
    INVESTIGATE = "investigate"  # Unusual pattern detected, needs review
    ALERT = "alert"  # Performance issue detected, notify user


@dataclass
class FeedbackAction:
    """Action to take based on performance analysis.

    Attributes:
        action: Type of action to take
        message: Human-readable explanation
        suggested_config: Recommended configuration (for rollback/update)
        severity: Urgency level (info, warning, error)
        metrics: Relevant metrics that triggered the action
    """

    action: ActionType
    message: str
    suggested_config: AnalyzerConfig | None = None
    severity: Literal["info", "warning", "error"] = "info"
    metrics: dict[str, float] = field(default_factory=dict)


class TrendDirection(str, Enum):
    """Direction of performance trend over time."""

    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"


class TrendReport(BaseModel):
    """Report on performance trends over time."""

    suite: str = Field(description="Test suite name")
    num_runs: int = Field(description="Number of runs analyzed")
    trend_direction: str = Field(description="Overall trend direction")
    trend_slope: float = Field(description="Slope of F-measure over time")
    volatility: float = Field(description="Standard deviation of F-measure")
    avg_f_measure: float = Field(description="Average F-measure")
    min_f_measure: float = Field(description="Minimum F-measure observed")
    max_f_measure: float = Field(description="Maximum F-measure observed")
    best_run_id: int | None = Field(description="ID of best performing run")
    best_run_date: str | None = Field(description="Date of best run")
    worst_run_id: int | None = Field(description="ID of worst performing run")
    worst_run_date: str | None = Field(description="Date of worst run")
    recommendation: str = Field(description="Suggested action based on trend")


class PerformanceMonitor:
    """Continuous monitoring with automatic feedback.

    Tracks performance over time, detects drift, and provides
    actionable feedback for maintaining optimal configurations.
    """

    def __init__(
        self,
        db: BenchmarkDB,
        alert_threshold: float = 0.03,
        rollback_threshold: float = 0.05,
        improvement_threshold: float = 0.02,
        volatility_threshold: float = 0.10,
        history_window: int = 10,
    ):
        """Initialize performance monitor.

        Args:
            db: BenchmarkDB instance for historical data
            alert_threshold: F-measure drop to trigger alert
            rollback_threshold: F-measure drop to trigger rollback
            improvement_threshold: F-measure improvement to update baseline
            volatility_threshold: Std dev threshold for volatility warning
            history_window: Number of recent runs to consider
        """
        self.db = db
        self.alert_threshold = alert_threshold
        self.rollback_threshold = rollback_threshold
        self.improvement_threshold = improvement_threshold
        self.volatility_threshold = volatility_threshold
        self.history_window = history_window

        # Cache for best known configurations per suite
        self._best_configs: dict[str, tuple[AnalyzerConfig, float]] = {}

    def record_and_evaluate(
        self,
        suite: str,
        results: dict[str, Any],
        aubio_version: str,
        current_config: AnalyzerConfig | None = None,
        commit_hash: str | None = None,
    ) -> FeedbackAction:
        """Record results and determine if action is needed.

        This is the main entry point for the feedback loop. Call after
        each benchmark run to get automatic recommendations.

        Args:
            suite: Test suite name
            results: Analysis results dictionary
            aubio_version: Aubio library version
            current_config: Current analyzer configuration
            commit_hash: Optional git commit hash

        Returns:
            FeedbackAction with recommended action
        """
        # Save current run
        run = self.db.save_run(suite, results, aubio_version, commit_hash)

        # Get historical data
        history = self.db.get_history(suite, limit=self.history_window)

        # Skip comparison if first run
        if len(history) <= 1:
            return FeedbackAction(
                action=ActionType.MAINTAIN,
                message="First run for this suite. Saving as baseline.",
                metrics={"f_measure": run.avg_f_measure},
            )

        # Compute rolling average (excluding current run)
        historical_runs = [h for h in history if h.id != run.id]
        if not historical_runs:
            return FeedbackAction(
                action=ActionType.MAINTAIN,
                message="Insufficient history for comparison.",
                metrics={"f_measure": run.avg_f_measure},
            )

        rolling_avg = np.mean([h.avg_f_measure for h in historical_runs])
        rolling_std = np.std([h.avg_f_measure for h in historical_runs])

        # Detect drift
        drift = run.avg_f_measure - rolling_avg

        # Determine action based on drift magnitude
        if drift < -self.rollback_threshold:
            # Significant regression - recommend rollback
            best_run = max(historical_runs, key=lambda r: r.avg_f_measure)
            best_config = self._get_config_from_run(best_run)

            return FeedbackAction(
                action=ActionType.ROLLBACK,
                message=(
                    f"Performance dropped significantly: {run.avg_f_measure:.3f} vs "
                    f"rolling avg {rolling_avg:.3f} (delta: {drift:.3f}). "
                    f"Recommend rollback to config from run {best_run.id}."
                ),
                suggested_config=best_config,
                severity="error",
                metrics={
                    "current_f_measure": run.avg_f_measure,
                    "rolling_avg": rolling_avg,
                    "drift": drift,
                    "best_f_measure": best_run.avg_f_measure,
                },
            )

        elif drift < -self.alert_threshold:
            # Moderate regression - alert but don't auto-rollback
            return FeedbackAction(
                action=ActionType.ALERT,
                message=(
                    f"Performance declined: {run.avg_f_measure:.3f} vs "
                    f"rolling avg {rolling_avg:.3f} (delta: {drift:.3f}). "
                    f"Consider investigating."
                ),
                severity="warning",
                metrics={
                    "current_f_measure": run.avg_f_measure,
                    "rolling_avg": rolling_avg,
                    "drift": drift,
                },
            )

        elif drift > self.improvement_threshold:
            # Improvement - update baseline
            return FeedbackAction(
                action=ActionType.UPDATE_BASELINE,
                message=(
                    f"Performance improved: {run.avg_f_measure:.3f} vs "
                    f"rolling avg {rolling_avg:.3f} (delta: +{drift:.3f}). "
                    f"Saving as new baseline."
                ),
                suggested_config=current_config,
                severity="info",
                metrics={
                    "current_f_measure": run.avg_f_measure,
                    "rolling_avg": rolling_avg,
                    "drift": drift,
                },
            )

        # Check for high volatility
        if rolling_std > self.volatility_threshold:
            return FeedbackAction(
                action=ActionType.INVESTIGATE,
                message=(
                    f"High performance volatility detected: σ={rolling_std:.3f}. "
                    f"Results may be unstable."
                ),
                severity="warning",
                metrics={
                    "current_f_measure": run.avg_f_measure,
                    "volatility": rolling_std,
                },
            )

        # Performance is stable
        return FeedbackAction(
            action=ActionType.MAINTAIN,
            message=f"Performance stable at {run.avg_f_measure:.3f} (±{rolling_std:.3f}).",
            metrics={
                "current_f_measure": run.avg_f_measure,
                "rolling_avg": rolling_avg,
                "volatility": rolling_std,
            },
        )

    def generate_trend_report(self, suite: str, limit: int = 20) -> TrendReport:
        """Analyze performance trends over time.

        Provides insights into whether performance is improving, stable,
        or declining, along with actionable recommendations.

        Args:
            suite: Test suite name
            limit: Maximum number of runs to analyze

        Returns:
            TrendReport with trend analysis and recommendations
        """
        history = self.db.get_history(suite, limit=limit)

        if not history:
            return TrendReport(
                suite=suite,
                num_runs=0,
                trend_direction=TrendDirection.STABLE.value,
                trend_slope=0.0,
                volatility=0.0,
                avg_f_measure=0.0,
                min_f_measure=0.0,
                max_f_measure=0.0,
                best_run_id=None,
                best_run_date=None,
                worst_run_id=None,
                worst_run_date=None,
                recommendation="No historical data available. Run benchmarks to establish baseline.",
            )

        # Extract F-measures (ordered by timestamp, newest first)
        f_measures = [run.avg_f_measure for run in history]

        # Compute statistics
        avg_f = np.mean(f_measures)
        min_f = np.min(f_measures)
        max_f = np.max(f_measures)
        volatility = np.std(f_measures)

        # Compute trend slope using linear regression
        # Note: history is newest first, so we reverse for proper slope
        x = np.arange(len(f_measures))
        f_measures_reversed = f_measures[::-1]
        if len(f_measures) > 1:
            slope = np.polyfit(x, f_measures_reversed, 1)[0]
        else:
            slope = 0.0

        # Determine trend direction
        if abs(slope) < 0.001:
            if volatility > self.volatility_threshold:
                direction = TrendDirection.VOLATILE
            else:
                direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.IMPROVING
        else:
            direction = TrendDirection.DECLINING

        # Find best and worst runs
        best_run = max(history, key=lambda r: r.avg_f_measure)
        worst_run = min(history, key=lambda r: r.avg_f_measure)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            direction, volatility, slope, f_measures[0], avg_f
        )

        return TrendReport(
            suite=suite,
            num_runs=len(history),
            trend_direction=direction.value,
            trend_slope=float(slope),
            volatility=float(volatility),
            avg_f_measure=float(avg_f),
            min_f_measure=float(min_f),
            max_f_measure=float(max_f),
            best_run_id=best_run.id,
            best_run_date=best_run.timestamp,
            worst_run_id=worst_run.id,
            worst_run_date=worst_run.timestamp,
            recommendation=recommendation,
        )

    def get_best_known_config(
        self, suite: str, refresh: bool = False
    ) -> tuple[AnalyzerConfig | None, float]:
        """Get the best known configuration for a suite.

        Args:
            suite: Test suite name
            refresh: Force refresh from database

        Returns:
            Tuple of (best_config, f_measure) or (None, 0.0) if not found
        """
        if not refresh and suite in self._best_configs:
            return self._best_configs[suite]

        history = self.db.get_history(suite, limit=50)
        if not history:
            return None, 0.0

        best_run = max(history, key=lambda r: r.avg_f_measure)
        config = self._get_config_from_run(best_run)

        if config:
            self._best_configs[suite] = (config, best_run.avg_f_measure)
            return config, best_run.avg_f_measure

        return None, best_run.avg_f_measure

    def check_for_regression(
        self, suite: str, current_f_measure: float
    ) -> tuple[bool, float]:
        """Quick check if current performance is a regression.

        Args:
            suite: Test suite name
            current_f_measure: Current F-measure

        Returns:
            Tuple of (is_regression, baseline_f_measure)
        """
        baseline = self.db.get_baseline(suite)
        if baseline is None:
            return False, current_f_measure

        delta = current_f_measure - baseline.avg_f_measure
        is_regression = delta < -self.alert_threshold

        return is_regression, baseline.avg_f_measure

    def _get_config_from_run(self, run: BenchmarkRun) -> AnalyzerConfig | None:
        """Extract analyzer config from benchmark run results.

        Args:
            run: BenchmarkRun to extract config from

        Returns:
            AnalyzerConfig if found, None otherwise
        """
        import json

        try:
            results = json.loads(run.results_json)
            # Look for config in results metadata
            if "config" in results:
                config_data = results["config"]
                return AnalyzerConfig(
                    fft_size=config_data.get("fft_size", 2048),
                    hop_size=config_data.get("hop_size", 512),
                    sample_rate=config_data.get("sample_rate", 44100),
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # Return default config if not found
        return AnalyzerConfig()

    def _generate_recommendation(
        self,
        direction: TrendDirection,
        volatility: float,
        slope: float,
        latest_f: float,
        avg_f: float,
    ) -> str:
        """Generate human-readable recommendation based on trend analysis."""
        if direction == TrendDirection.IMPROVING:
            return (
                f"Performance trending upward (slope: +{slope:.4f}/run). "
                f"Current approach is working well. Consider continuing optimization."
            )
        elif direction == TrendDirection.DECLINING:
            return (
                f"Performance trending downward (slope: {slope:.4f}/run). "
                f"Review recent changes and consider rollback to best known config."
            )
        elif direction == TrendDirection.VOLATILE:
            return (
                f"High volatility detected (σ={volatility:.3f}). "
                f"Results may be inconsistent. Consider using fixed test signals and "
                f"reviewing for environmental factors."
            )
        else:  # STABLE
            if avg_f >= 0.9:
                return (
                    f"Performance stable at excellent level ({avg_f:.3f}). "
                    f"Focus on maintaining current configuration."
                )
            elif avg_f >= 0.7:
                return (
                    f"Performance stable at good level ({avg_f:.3f}). "
                    f"Consider parameter tuning to improve further."
                )
            else:
                return (
                    f"Performance stable but low ({avg_f:.3f}). "
                    f"Significant optimization needed. Consider grid search for "
                    f"better parameters."
                )


class FeedbackLoop:
    """High-level interface for closed-loop optimization.

    Combines performance monitoring, trend analysis, and automatic
    configuration management into a single easy-to-use interface.
    """

    def __init__(self, db_path: str = "benchmarks.db"):
        """Initialize feedback loop.

        Args:
            db_path: Path to SQLite benchmark database
        """
        self.db = BenchmarkDB(db_path)
        self.monitor = PerformanceMonitor(self.db)
        self._active_configs: dict[str, AnalyzerConfig] = {}

    def run_with_feedback(
        self,
        suite: str,
        results: dict[str, Any],
        aubio_version: str,
        config: AnalyzerConfig | None = None,
    ) -> FeedbackAction:
        """Record run results and get feedback.

        Args:
            suite: Test suite name
            results: Analysis results
            aubio_version: Aubio version string
            config: Current configuration

        Returns:
            FeedbackAction with recommended next steps
        """
        if config:
            self._active_configs[suite] = config

        action = self.monitor.record_and_evaluate(
            suite=suite,
            results=results,
            aubio_version=aubio_version,
            current_config=config,
        )

        # Handle automatic actions
        if action.action == ActionType.UPDATE_BASELINE and config:
            self._active_configs[suite] = config
        elif action.action == ActionType.ROLLBACK and action.suggested_config:
            self._active_configs[suite] = action.suggested_config

        return action

    def get_current_config(self, suite: str) -> AnalyzerConfig:
        """Get current active configuration for a suite.

        Args:
            suite: Test suite name

        Returns:
            Current AnalyzerConfig (or default if not set)
        """
        return self._active_configs.get(suite, AnalyzerConfig())

    def get_trend_report(self, suite: str) -> TrendReport:
        """Get trend analysis for a suite.

        Args:
            suite: Test suite name

        Returns:
            TrendReport with analysis
        """
        return self.monitor.generate_trend_report(suite)

    def get_all_suite_status(self) -> dict[str, dict[str, Any]]:
        """Get quick status summary for all known suites.

        Returns:
            Dictionary mapping suite names to status info
        """
        status = {}

        # Get unique suites from history
        with self.db.db_path.open():
            import sqlite3

            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.execute("SELECT DISTINCT suite FROM benchmark_runs")
            suites = [row[0] for row in cursor.fetchall()]
            conn.close()

        for suite in suites:
            baseline = self.db.get_baseline(suite)
            if baseline:
                config = self._active_configs.get(suite)
                status[suite] = {
                    "latest_f_measure": baseline.avg_f_measure,
                    "latest_timestamp": baseline.timestamp,
                    "aubio_version": baseline.aubio_version,
                    "has_active_config": config is not None,
                }

        return status
