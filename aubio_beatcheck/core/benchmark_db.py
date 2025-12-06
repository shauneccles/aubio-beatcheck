"""Benchmark Database for Regression Tracking.

SQLite-backed storage for benchmark results to track performance
across aubio versions and detect regressions.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class BenchmarkRun(BaseModel):
    """A single benchmark run with metadata."""

    id: int | None = Field(default=None, description="Database ID")
    timestamp: str = Field(description="ISO timestamp of the run")
    aubio_version: str = Field(description="Aubio library version")
    commit_hash: str | None = Field(default=None, description="Git commit hash")
    suite: str = Field(description="Test suite name")
    total_signals: int = Field(description="Number of signals tested")
    completed: int = Field(description="Number successfully completed")
    failed: int = Field(description="Number that failed")
    avg_precision: float = Field(description="Average precision across all signals")
    avg_recall: float = Field(description="Average recall across all signals")
    avg_f_measure: float = Field(description="Average F-measure")
    avg_mae_ms: float = Field(description="Average mean absolute error in ms")
    avg_processing_time_ms: float = Field(
        description="Average processing time per signal"
    )
    results_json: str = Field(description="Full results as JSON string")


class RegressionReport(BaseModel):
    """Report comparing current results to baseline."""

    is_regression: bool = Field(description="True if regression detected")
    baseline_run: BenchmarkRun | None = Field(description="Baseline run for comparison")
    current_run: BenchmarkRun = Field(description="Current run")
    f_measure_delta: float = Field(description="Change in F-measure (negative = worse)")
    precision_delta: float = Field(description="Change in precision")
    recall_delta: float = Field(description="Change in recall")
    mae_delta_ms: float = Field(description="Change in MAE (positive = worse)")
    processing_time_delta_ms: float = Field(description="Change in processing time")
    regression_threshold: float = Field(
        default=0.05, description="Threshold for regression"
    )
    details: str = Field(default="", description="Human-readable summary")


class BenchmarkDB:
    """SQLite database for storing benchmark results.

    Provides methods to save runs, query history, and detect regressions.
    """

    def __init__(self, db_path: Path | str = "benchmarks.db"):
        """Initialize the benchmark database.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    aubio_version TEXT NOT NULL,
                    commit_hash TEXT,
                    suite TEXT NOT NULL,
                    total_signals INTEGER NOT NULL,
                    completed INTEGER NOT NULL,
                    failed INTEGER NOT NULL,
                    avg_precision REAL NOT NULL,
                    avg_recall REAL NOT NULL,
                    avg_f_measure REAL NOT NULL,
                    avg_mae_ms REAL NOT NULL,
                    avg_processing_time_ms REAL NOT NULL,
                    results_json TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_suite_timestamp
                ON benchmark_runs (suite, timestamp DESC)
            """)
            conn.commit()

    def save_run(
        self,
        suite: str,
        results: dict[str, Any],
        aubio_version: str,
        commit_hash: str | None = None,
    ) -> BenchmarkRun:
        """Save a benchmark run to the database.

        Args:
            suite: Test suite name.
            results: Analysis results dictionary.
            aubio_version: Aubio library version string.
            commit_hash: Git commit hash (optional).

        Returns:
            BenchmarkRun with the saved data including ID.
        """
        # Calculate aggregates
        signals = results.get("signals", [])
        completed_signals = [s for s in signals if s.get("status") == "completed"]

        avg_precision = 0.0
        avg_recall = 0.0
        avg_f_measure = 0.0
        avg_mae_ms = 0.0
        avg_processing_time_ms = 0.0

        if completed_signals:
            precisions = []
            recalls = []
            f_measures = []
            maes = []
            times = []

            for s in completed_signals:
                eval_data = s.get("evaluation")
                if eval_data:
                    precisions.append(eval_data.get("precision", 0))
                    recalls.append(eval_data.get("recall", 0))
                    f_measures.append(eval_data.get("f_measure", 0))
                    maes.append(eval_data.get("mean_absolute_error_ms", 0))
                if s.get("performance_mean_us"):
                    times.append(s["performance_mean_us"] / 1000)  # Convert to ms

            if precisions:
                avg_precision = sum(precisions) / len(precisions)
                avg_recall = sum(recalls) / len(recalls)
                avg_f_measure = sum(f_measures) / len(f_measures)
                avg_mae_ms = sum(maes) / len(maes)
            if times:
                avg_processing_time_ms = sum(times) / len(times)

        run = BenchmarkRun(
            timestamp=datetime.utcnow().isoformat(),
            aubio_version=aubio_version,
            commit_hash=commit_hash,
            suite=suite,
            total_signals=len(signals),
            completed=len(completed_signals),
            failed=len(signals) - len(completed_signals),
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_f_measure=avg_f_measure,
            avg_mae_ms=avg_mae_ms,
            avg_processing_time_ms=avg_processing_time_ms,
            results_json=json.dumps(results),
        )

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO benchmark_runs (
                    timestamp, aubio_version, commit_hash, suite,
                    total_signals, completed, failed,
                    avg_precision, avg_recall, avg_f_measure,
                    avg_mae_ms, avg_processing_time_ms, results_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    run.timestamp,
                    run.aubio_version,
                    run.commit_hash,
                    run.suite,
                    run.total_signals,
                    run.completed,
                    run.failed,
                    run.avg_precision,
                    run.avg_recall,
                    run.avg_f_measure,
                    run.avg_mae_ms,
                    run.avg_processing_time_ms,
                    run.results_json,
                ),
            )
            run.id = cursor.lastrowid
            conn.commit()

        return run

    def get_baseline(self, suite: str) -> BenchmarkRun | None:
        """Get the most recent baseline run for a suite.

        Args:
            suite: Test suite name.

        Returns:
            Most recent BenchmarkRun or None if no baseline exists.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM benchmark_runs
                WHERE suite = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """,
                (suite,),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return BenchmarkRun(**dict(row))

    def compare_to_baseline(
        self,
        current: BenchmarkRun,
        regression_threshold: float = 0.05,
    ) -> RegressionReport:
        """Compare current run to baseline and detect regressions.

        Args:
            current: Current benchmark run.
            regression_threshold: F-measure drop threshold for regression.

        Returns:
            RegressionReport with comparison details.
        """
        baseline = self.get_baseline(current.suite)

        if baseline is None:
            return RegressionReport(
                is_regression=False,
                baseline_run=None,
                current_run=current,
                f_measure_delta=0.0,
                precision_delta=0.0,
                recall_delta=0.0,
                mae_delta_ms=0.0,
                processing_time_delta_ms=0.0,
                regression_threshold=regression_threshold,
                details="No baseline available for comparison.",
            )

        f_delta = current.avg_f_measure - baseline.avg_f_measure
        p_delta = current.avg_precision - baseline.avg_precision
        r_delta = current.avg_recall - baseline.avg_recall
        mae_delta = current.avg_mae_ms - baseline.avg_mae_ms
        time_delta = current.avg_processing_time_ms - baseline.avg_processing_time_ms

        is_regression = f_delta < -regression_threshold

        details = f"F-measure: {baseline.avg_f_measure:.3f} → {current.avg_f_measure:.3f} ({f_delta:+.3f})"
        if is_regression:
            details = f"⚠️ REGRESSION DETECTED: {details}"

        return RegressionReport(
            is_regression=is_regression,
            baseline_run=baseline,
            current_run=current,
            f_measure_delta=f_delta,
            precision_delta=p_delta,
            recall_delta=r_delta,
            mae_delta_ms=mae_delta,
            processing_time_delta_ms=time_delta,
            regression_threshold=regression_threshold,
            details=details,
        )

    def get_history(
        self,
        suite: str,
        limit: int = 10,
    ) -> list[BenchmarkRun]:
        """Get historical benchmark runs for a suite.

        Args:
            suite: Test suite name.
            limit: Maximum number of runs to return.

        Returns:
            List of BenchmarkRun objects, most recent first.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM benchmark_runs
                WHERE suite = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (suite, limit),
            )
            rows = cursor.fetchall()

        return [BenchmarkRun(**dict(row)) for row in rows]
