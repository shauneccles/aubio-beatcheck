"""Parameter Grid Search for Aubio Optimization.

Systematically test aubio parameters to find optimal configurations
for specific signals or use cases.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from aubio_beatcheck.core.analyzers import AnalyzerConfig, OnsetAnalyzer, TempoAnalyzer
from aubio_beatcheck.core.evaluation import Evaluator


class GridSearchResult(BaseModel):
    """Result from a single parameter combination."""

    params: dict[str, Any] = Field(description="Parameter combination tested")
    f_measure: float = Field(description="F-measure score")
    precision: float = Field(description="Precision score")
    recall: float = Field(description="Recall score")
    mae_ms: float = Field(description="Mean absolute error in ms")
    processing_time_ms: float = Field(description="Total processing time in ms")


class GridSearchResults(BaseModel):
    """Complete results from a grid search."""

    best_params: dict[str, Any] = Field(description="Best parameter combination")
    best_score: float = Field(description="Best F-measure score")
    all_results: list[GridSearchResult] = Field(description="All tested combinations")
    total_combinations: int = Field(description="Total combinations tested")


@dataclass
class ParameterGrid:
    """Defines a grid of parameters to search over.

    Attributes:
        fft_sizes: List of FFT sizes to test.
        hop_sizes: List of hop sizes to test.
        thresholds: List of detection thresholds to test.
    """

    fft_sizes: list[int] = field(default_factory=lambda: [1024, 2048, 4096])
    hop_sizes: list[int] = field(default_factory=lambda: [256, 512, 1024])
    thresholds: list[float] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5])

    def __len__(self) -> int:
        """Return total number of combinations."""
        return len(self.fft_sizes) * len(self.hop_sizes) * len(self.thresholds)

    def __iter__(self):
        """Iterate over all parameter combinations."""
        for fft, hop, thresh in product(
            self.fft_sizes, self.hop_sizes, self.thresholds
        ):
            yield {"fft_size": fft, "hop_size": hop, "threshold": thresh}


def search_tempo_params(
    audio: np.ndarray,
    ground_truth_beats: list[float],
    sample_rate: int = 44100,
    tolerance_ms: float = 50.0,
    grid: ParameterGrid | None = None,
) -> GridSearchResults:
    """Search for optimal tempo detection parameters.

    Args:
        audio: Audio samples as float32 array.
        ground_truth_beats: List of ground truth beat times in seconds.
        sample_rate: Audio sample rate.
        tolerance_ms: Timing tolerance for evaluation.
        grid: Parameter grid to search (uses defaults if None).

    Returns:
        GridSearchResults with best parameters and all results.
    """
    if grid is None:
        grid = ParameterGrid()

    results: list[GridSearchResult] = []
    best_result: GridSearchResult | None = None

    for params in grid:
        config = AnalyzerConfig(
            fft_size=params["fft_size"],
            hop_size=params["hop_size"],
            sample_rate=sample_rate,
        )

        analyzer = TempoAnalyzer(config)

        import time

        t0 = time.perf_counter()
        detected_beats, _ = analyzer.analyze(audio)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        metrics = Evaluator.evaluate_events(
            detected_beats, ground_truth_beats, tolerance_ms=tolerance_ms
        )

        result = GridSearchResult(
            params=params,
            f_measure=metrics.f_measure,
            precision=metrics.precision,
            recall=metrics.recall,
            mae_ms=metrics.mean_absolute_error_ms,
            processing_time_ms=elapsed_ms,
        )
        results.append(result)

        if best_result is None or result.f_measure > best_result.f_measure:
            best_result = result

    return GridSearchResults(
        best_params=best_result.params if best_result else {},
        best_score=best_result.f_measure if best_result else 0.0,
        all_results=results,
        total_combinations=len(grid),
    )


def search_onset_params(
    audio: np.ndarray,
    ground_truth_onsets: list[float],
    sample_rate: int = 44100,
    tolerance_ms: float = 50.0,
    grid: ParameterGrid | None = None,
) -> GridSearchResults:
    """Search for optimal onset detection parameters.

    Args:
        audio: Audio samples as float32 array.
        ground_truth_onsets: List of ground truth onset times in seconds.
        sample_rate: Audio sample rate.
        tolerance_ms: Timing tolerance for evaluation.
        grid: Parameter grid to search (uses defaults if None).

    Returns:
        GridSearchResults with best parameters and all results.
    """
    if grid is None:
        grid = ParameterGrid()

    results: list[GridSearchResult] = []
    best_result: GridSearchResult | None = None

    for params in grid:
        config = AnalyzerConfig(
            fft_size=params["fft_size"],
            hop_size=params["hop_size"],
            sample_rate=sample_rate,
        )

        analyzer = OnsetAnalyzer(config, threshold=params["threshold"])

        import time

        t0 = time.perf_counter()
        detected_onsets = analyzer.analyze(audio)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        metrics = Evaluator.evaluate_events(
            detected_onsets, ground_truth_onsets, tolerance_ms=tolerance_ms
        )

        result = GridSearchResult(
            params=params,
            f_measure=metrics.f_measure,
            precision=metrics.precision,
            recall=metrics.recall,
            mae_ms=metrics.mean_absolute_error_ms,
            processing_time_ms=elapsed_ms,
        )
        results.append(result)

        if best_result is None or result.f_measure > best_result.f_measure:
            best_result = result

    return GridSearchResults(
        best_params=best_result.params if best_result else {},
        best_score=best_result.f_measure if best_result else 0.0,
        all_results=results,
        total_combinations=len(grid),
    )
