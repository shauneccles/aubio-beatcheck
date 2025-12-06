"""Multi-Objective Optimization for Aubio Parameter Tuning.

Provides weighted multi-metric optimization and Pareto-optimal configuration
finding for closed-loop tuning of aubio analysis parameters.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from aubio_beatcheck.core.analyzers import PerformanceStats
from aubio_beatcheck.core.evaluation import EvaluationMetrics
from aubio_beatcheck.core.grid_search import GridSearchResult


@dataclass
class OptimizationObjectives:
    """Weighted objectives for closed-loop tuning.

    Weights should sum to 1.0 for normalized composite scores.
    Adjust weights based on application requirements.

    Attributes:
        precision_weight: Weight for precision (avoiding false positives)
        recall_weight: Weight for recall (avoiding missed events)
        timing_accuracy_weight: Weight for timing accuracy (inverse of MAE)
        latency_weight: Weight for low latency (real-time applications)
    """

    precision_weight: float = 0.25
    recall_weight: float = 0.25
    timing_accuracy_weight: float = 0.25
    latency_weight: float = 0.25

    def __post_init__(self):
        """Normalize weights to sum to 1.0."""
        total = (
            self.precision_weight
            + self.recall_weight
            + self.timing_accuracy_weight
            + self.latency_weight
        )
        if total > 0:
            self.precision_weight /= total
            self.recall_weight /= total
            self.timing_accuracy_weight /= total
            self.latency_weight /= total

    @classmethod
    def precision_focused(cls) -> "OptimizationObjectives":
        """Objectives focused on precision (minimizing false positives)."""
        return cls(
            precision_weight=0.5,
            recall_weight=0.2,
            timing_accuracy_weight=0.2,
            latency_weight=0.1,
        )

    @classmethod
    def recall_focused(cls) -> "OptimizationObjectives":
        """Objectives focused on recall (minimizing missed events)."""
        return cls(
            precision_weight=0.2,
            recall_weight=0.5,
            timing_accuracy_weight=0.2,
            latency_weight=0.1,
        )

    @classmethod
    def timing_focused(cls) -> "OptimizationObjectives":
        """Objectives focused on timing accuracy."""
        return cls(
            precision_weight=0.15,
            recall_weight=0.15,
            timing_accuracy_weight=0.6,
            latency_weight=0.1,
        )

    @classmethod
    def realtime_focused(cls) -> "OptimizationObjectives":
        """Objectives for real-time applications (low latency priority)."""
        return cls(
            precision_weight=0.2,
            recall_weight=0.2,
            timing_accuracy_weight=0.2,
            latency_weight=0.4,
        )

    @classmethod
    def balanced(cls) -> "OptimizationObjectives":
        """Balanced objectives (equal weights)."""
        return cls()


class CompositeScore(BaseModel):
    """Composite score with breakdown of individual metric contributions."""

    total_score: float = Field(description="Weighted composite score (0-1)")
    precision_contribution: float = Field(description="Precision component")
    recall_contribution: float = Field(description="Recall component")
    timing_contribution: float = Field(description="Timing accuracy component")
    latency_contribution: float = Field(description="Latency component")
    raw_metrics: dict[str, float] = Field(description="Raw metric values")


class MultiObjectiveEvaluator:
    """Compute composite scores for multi-objective optimization.

    Combines multiple metrics into a single scalar score using weighted
    objectives, enabling optimization algorithms to compare configurations.
    """

    # Normalization constants for unbounded metrics
    MAE_NORMALIZATION_MS: float = 50.0  # MAE at which timing score = 0.5
    LATENCY_NORMALIZATION_US: float = 1000.0  # Latency at which score = 0.5

    def __init__(self, objectives: OptimizationObjectives | None = None):
        """Initialize evaluator with optimization objectives.

        Args:
            objectives: Weighted objectives (uses balanced if None)
        """
        self.objectives = objectives or OptimizationObjectives.balanced()

    def compute_composite_score(
        self,
        metrics: EvaluationMetrics,
        performance_stats: PerformanceStats | None = None,
    ) -> CompositeScore:
        """Compute weighted composite score from metrics.

        Args:
            metrics: Evaluation metrics from analysis
            performance_stats: Optional performance statistics for latency

        Returns:
            CompositeScore with total and component breakdown
        """
        # Timing accuracy score (inverse of MAE, normalized)
        # Score of 1.0 at 0ms error, 0.5 at MAE_NORMALIZATION_MS
        timing_score = 1.0 / (
            1.0 + metrics.mean_absolute_error_ms / self.MAE_NORMALIZATION_MS
        )

        # Latency score (inverse of P95, normalized)
        if performance_stats and performance_stats.p95_us > 0:
            latency_score = 1.0 / (
                1.0 + performance_stats.p95_us / self.LATENCY_NORMALIZATION_US
            )
        else:
            latency_score = 1.0  # Assume perfect if not measured

        # Compute weighted contributions
        precision_contrib = self.objectives.precision_weight * metrics.precision
        recall_contrib = self.objectives.recall_weight * metrics.recall
        timing_contrib = self.objectives.timing_accuracy_weight * timing_score
        latency_contrib = self.objectives.latency_weight * latency_score

        total = precision_contrib + recall_contrib + timing_contrib + latency_contrib

        return CompositeScore(
            total_score=total,
            precision_contribution=precision_contrib,
            recall_contribution=recall_contrib,
            timing_contribution=timing_contrib,
            latency_contribution=latency_contrib,
            raw_metrics={
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f_measure": metrics.f_measure,
                "mae_ms": metrics.mean_absolute_error_ms,
                "timing_score": timing_score,
                "latency_score": latency_score,
                "p95_us": performance_stats.p95_us if performance_stats else 0.0,
            },
        )

    def compare_configurations(
        self,
        results: list[tuple[dict[str, Any], EvaluationMetrics, PerformanceStats | None]],
    ) -> list[tuple[dict[str, Any], CompositeScore]]:
        """Compare multiple configurations and rank by composite score.

        Args:
            results: List of (params, metrics, stats) tuples

        Returns:
            Sorted list of (params, score) tuples, best first
        """
        scored = []
        for params, metrics, stats in results:
            score = self.compute_composite_score(metrics, stats)
            scored.append((params, score))

        # Sort by total score, descending
        scored.sort(key=lambda x: x[1].total_score, reverse=True)
        return scored


@dataclass
class ParetoPoint:
    """A point in the Pareto front with its parameter configuration."""

    params: dict[str, Any]
    precision: float
    recall: float
    timing_score: float
    latency_score: float
    f_measure: float
    mae_ms: float

    def dominates(self, other: "ParetoPoint") -> bool:
        """Check if this point dominates another (better in all objectives)."""
        better_in_all = (
            self.precision >= other.precision
            and self.recall >= other.recall
            and self.timing_score >= other.timing_score
            and self.latency_score >= other.latency_score
        )
        better_in_one = (
            self.precision > other.precision
            or self.recall > other.recall
            or self.timing_score > other.timing_score
            or self.latency_score > other.latency_score
        )
        return better_in_all and better_in_one


class ParetoOptimizer:
    """Find Pareto-optimal configurations across multiple objectives.

    Pareto-optimal configurations are those where no other configuration
    is better in all objectives simultaneously. This allows decision-makers
    to choose based on their specific tradeoff preferences.
    """

    MAE_NORMALIZATION_MS: float = 50.0
    LATENCY_NORMALIZATION_US: float = 1000.0

    def find_pareto_front(
        self,
        results: list[GridSearchResult],
        performance_stats: list[PerformanceStats] | None = None,
    ) -> list[ParetoPoint]:
        """Find Pareto-optimal configurations from grid search results.

        Args:
            results: Grid search results
            performance_stats: Optional matching performance statistics

        Returns:
            List of non-dominated ParetoPoint configurations
        """
        # Convert results to ParetoPoints
        points = []
        for i, result in enumerate(results):
            timing_score = 1.0 / (1.0 + result.mae_ms / self.MAE_NORMALIZATION_MS)

            if performance_stats and i < len(performance_stats):
                p95 = performance_stats[i].p95_us
                latency_score = 1.0 / (1.0 + p95 / self.LATENCY_NORMALIZATION_US)
            else:
                # Estimate from processing time if available
                # Assume processing_time_ms correlates with frame latency
                latency_score = 1.0 / (
                    1.0 + result.processing_time_ms * 1000 / self.LATENCY_NORMALIZATION_US
                )

            points.append(
                ParetoPoint(
                    params=result.params,
                    precision=result.precision,
                    recall=result.recall,
                    timing_score=timing_score,
                    latency_score=latency_score,
                    f_measure=result.f_measure,
                    mae_ms=result.mae_ms,
                )
            )

        # Find non-dominated points
        pareto_front = []
        for candidate in points:
            is_dominated = False
            for other in points:
                if other is not candidate and other.dominates(candidate):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(candidate)

        return pareto_front

    def select_from_pareto(
        self, pareto_front: list[ParetoPoint], objectives: OptimizationObjectives
    ) -> ParetoPoint | None:
        """Select best configuration from Pareto front based on objectives.

        Args:
            pareto_front: Pareto-optimal configurations
            objectives: Weighted optimization objectives

        Returns:
            Best configuration according to weighted objectives
        """
        if not pareto_front:
            return None

        best_point = None
        best_score = -1.0

        for point in pareto_front:
            score = (
                objectives.precision_weight * point.precision
                + objectives.recall_weight * point.recall
                + objectives.timing_accuracy_weight * point.timing_score
                + objectives.latency_weight * point.latency_score
            )
            if score > best_score:
                best_score = score
                best_point = point

        return best_point

    def summarize_pareto_front(self, pareto_front: list[ParetoPoint]) -> dict[str, Any]:
        """Generate summary statistics for Pareto front.

        Args:
            pareto_front: Pareto-optimal configurations

        Returns:
            Dictionary with summary statistics
        """
        if not pareto_front:
            return {"count": 0}

        precisions = [p.precision for p in pareto_front]
        recalls = [p.recall for p in pareto_front]
        f_measures = [p.f_measure for p in pareto_front]
        mae_values = [p.mae_ms for p in pareto_front]

        return {
            "count": len(pareto_front),
            "precision_range": (min(precisions), max(precisions)),
            "recall_range": (min(recalls), max(recalls)),
            "f_measure_range": (min(f_measures), max(f_measures)),
            "mae_range_ms": (min(mae_values), max(mae_values)),
            "best_precision_params": pareto_front[
                np.argmax(precisions)
            ].params,
            "best_recall_params": pareto_front[np.argmax(recalls)].params,
            "best_f_measure_params": pareto_front[
                np.argmax(f_measures)
            ].params,
            "best_timing_params": pareto_front[
                np.argmin(mae_values)
            ].params,
        }


class ObjectivePresets:
    """Pre-defined optimization objective configurations for common use cases."""

    PRESETS: dict[str, OptimizationObjectives] = {
        "balanced": OptimizationObjectives.balanced(),
        "precision": OptimizationObjectives.precision_focused(),
        "recall": OptimizationObjectives.recall_focused(),
        "timing": OptimizationObjectives.timing_focused(),
        "realtime": OptimizationObjectives.realtime_focused(),
    }

    @classmethod
    def get_preset(cls, name: str) -> OptimizationObjectives:
        """Get optimization objectives by preset name.

        Args:
            name: Preset name (balanced, precision, recall, timing, realtime)

        Returns:
            OptimizationObjectives configuration

        Raises:
            KeyError: If preset name not found
        """
        if name not in cls.PRESETS:
            available = ", ".join(cls.PRESETS.keys())
            raise KeyError(f"Unknown preset '{name}'. Available: {available}")
        return cls.PRESETS[name]

    @classmethod
    def list_presets(cls) -> dict[str, dict[str, float]]:
        """List all available objective presets with their weights.

        Returns:
            Dictionary mapping preset names to weight dictionaries
        """
        return {
            name: {
                "precision_weight": obj.precision_weight,
                "recall_weight": obj.recall_weight,
                "timing_accuracy_weight": obj.timing_accuracy_weight,
                "latency_weight": obj.latency_weight,
            }
            for name, obj in cls.PRESETS.items()
        }
