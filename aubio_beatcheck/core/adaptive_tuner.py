"""Adaptive Parameter Tuning Engine.

Provides learning-based parameter optimization using historical benchmark
data to find and maintain optimal configurations for aubio analysis.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from aubio_beatcheck.core.analyzers import AnalyzerConfig
from aubio_beatcheck.core.benchmark_db import BenchmarkDB
from aubio_beatcheck.core.evaluation import Evaluator
from aubio_beatcheck.core.multi_objective import (
    MultiObjectiveEvaluator,
    OptimizationObjectives,
    ParetoOptimizer,
)


class TuningHistory(BaseModel):
    """Historical record of parameter tuning attempts."""

    suite: str = Field(description="Test suite name")
    params: dict[str, Any] = Field(description="Parameters tested")
    f_measure: float = Field(description="Achieved F-measure")
    precision: float = Field(description="Precision")
    recall: float = Field(description="Recall")
    mae_ms: float = Field(description="Mean absolute error in ms")
    timestamp: str = Field(description="ISO timestamp")
    aubio_version: str = Field(description="Aubio version used")
    composite_score: float = Field(default=0.0, description="Weighted composite score")


class LearnedPreset(BaseModel):
    """A preset configuration learned from optimization."""

    name: str = Field(description="Preset name")
    suite: str = Field(description="Source test suite")
    config: dict[str, int] = Field(description="Analyzer configuration")
    f_measure: float = Field(description="Best F-measure achieved")
    composite_score: float = Field(description="Best composite score")
    num_trials: int = Field(description="Number of optimization trials")
    last_updated: str = Field(description="Last update timestamp")
    confidence: float = Field(description="Confidence in this preset (0-1)")


class SuggestionResult(BaseModel):
    """Result of parameter suggestion with confidence."""

    params: dict[str, Any] = Field(description="Suggested parameters")
    expected_score: float = Field(description="Expected composite score")
    confidence: float = Field(description="Confidence in suggestion (0-1)")
    exploration_weight: float = Field(description="Exploration vs exploitation ratio")
    reasoning: str = Field(description="Why these parameters were suggested")


class AdaptiveTuner:
    """Adaptive parameter tuning with learning from historical data.

    Uses a simplified Bayesian-like approach to:
    1. Learn from historical benchmark results
    2. Suggest promising parameter combinations
    3. Balance exploration vs exploitation
    4. Export optimized presets
    """

    def __init__(
        self,
        db: BenchmarkDB,
        objectives: OptimizationObjectives | None = None,
        exploration_rate: float = 0.2,
        min_trials_for_confidence: int = 10,
    ):
        """Initialize adaptive tuner.

        Args:
            db: BenchmarkDB instance for historical data
            objectives: Optimization objectives (uses balanced if None)
            exploration_rate: Probability of exploration vs exploitation
            min_trials_for_confidence: Minimum trials for high confidence
        """
        self.db = db
        self.objectives = objectives or OptimizationObjectives.balanced()
        self.exploration_rate = exploration_rate
        self.min_trials_for_confidence = min_trials_for_confidence

        self.evaluator = MultiObjectiveEvaluator(self.objectives)
        self.pareto = ParetoOptimizer()

        # Cache for learned parameters
        self._parameter_stats: dict[str, dict[str, Any]] = {}
        self._best_configs: dict[str, LearnedPreset] = {}

    def suggest(
        self,
        suite: str,
        signal_type: str = "tempo",
        exclude_params: list[dict[str, Any]] | None = None,
    ) -> SuggestionResult:
        """Suggest next parameters to try based on history.

        Uses a Thompson Sampling-inspired approach:
        1. Model each parameter's effect on score
        2. Sample from posterior based on uncertainty
        3. Balance exploration (uncertain) vs exploitation (known good)

        Args:
            suite: Test suite name
            signal_type: Type of signal (tempo, onset, pitch)
            exclude_params: Parameter combinations to exclude

        Returns:
            SuggestionResult with suggested parameters
        """
        exclude_params = exclude_params or []

        # Load and analyze history
        stats = self._analyze_history(suite)

        if not stats["has_data"]:
            # No history - suggest exploration with default grid middle
            return SuggestionResult(
                params={"fft_size": 2048, "hop_size": 512, "threshold": 0.3},
                expected_score=0.5,
                confidence=0.1,
                exploration_weight=1.0,
                reasoning="No historical data. Starting with default parameters.",
            )

        # Decide: explore or exploit?
        rng = np.random.default_rng()
        should_explore = rng.random() < self.exploration_rate

        if should_explore:
            return self._suggest_exploration(stats, exclude_params)
        else:
            return self._suggest_exploitation(stats, exclude_params)

    def update(
        self,
        suite: str,
        params: dict[str, Any],
        f_measure: float,
        precision: float,
        recall: float,
        mae_ms: float,
        aubio_version: str = "unknown",
    ) -> None:
        """Update tuning history with new result.

        Args:
            suite: Test suite name
            params: Parameters that were tested
            f_measure: Achieved F-measure
            precision: Precision
            recall: Recall
            mae_ms: Mean absolute error
            aubio_version: Aubio version
        """
        # Invalidate cache
        if suite in self._parameter_stats:
            del self._parameter_stats[suite]

        # Store in internal tracking (the actual DB storage happens via save_run)
        # Here we just update our analysis

    def tune(
        self,
        audio: np.ndarray,
        ground_truth: list[float],
        suite: str,
        signal_type: str = "tempo",
        max_iterations: int = 10,
        tolerance_ms: float = 50.0,
    ) -> tuple[AnalyzerConfig, float]:
        """Run adaptive tuning to find optimal parameters.

        Args:
            audio: Audio samples
            ground_truth: Ground truth event times
            suite: Test suite name for tracking
            signal_type: Type of analysis
            max_iterations: Maximum tuning iterations
            tolerance_ms: Timing tolerance for evaluation

        Returns:
            Tuple of (best_config, best_score)
        """
        from aubio_beatcheck.core.analyzers import OnsetAnalyzer, TempoAnalyzer

        best_config = None
        best_score = 0.0
        tested_params = []

        for _iteration in range(max_iterations):
            # Get suggestion
            suggestion = self.suggest(suite, signal_type, tested_params)
            params = suggestion.params

            tested_params.append(params)

            # Build config and analyzer
            config = AnalyzerConfig(
                fft_size=params.get("fft_size", 2048),
                hop_size=params.get("hop_size", 512),
                sample_rate=44100,
            )

            # Run analysis
            if signal_type == "tempo":
                analyzer = TempoAnalyzer(config)
                detected, _ = analyzer.analyze(audio)
            else:
                threshold = params.get("threshold", 0.3)
                analyzer = OnsetAnalyzer(config, threshold=threshold)
                detected = analyzer.analyze(audio)

            # Evaluate
            metrics = Evaluator.evaluate_events(detected, ground_truth, tolerance_ms)

            # Compute composite score

            stats = analyzer.stats
            composite = self.evaluator.compute_composite_score(metrics, stats)

            # Update tracking
            self.update(
                suite=suite,
                params=params,
                f_measure=metrics.f_measure,
                precision=metrics.precision,
                recall=metrics.recall,
                mae_ms=metrics.mean_absolute_error_ms,
            )

            # Track best
            if composite.total_score > best_score:
                best_score = composite.total_score
                best_config = config

            # Early stopping if we find excellent result
            if composite.total_score > 0.95:
                break

        return best_config or AnalyzerConfig(), best_score

    def get_best_config(self, suite: str) -> LearnedPreset | None:
        """Get the best learned configuration for a suite.

        Args:
            suite: Test suite name

        Returns:
            LearnedPreset or None if not found
        """
        if suite in self._best_configs:
            return self._best_configs[suite]

        # Compute from history
        stats = self._analyze_history(suite)
        if not stats["has_data"]:
            return None

        best_params = stats["best_params"]
        best_score = stats["best_score"]

        preset = LearnedPreset(
            name=f"{suite}_optimized",
            suite=suite,
            config={
                "fft_size": best_params.get("fft_size", 2048),
                "hop_size": best_params.get("hop_size", 512),
                "sample_rate": 44100,
            },
            f_measure=stats["best_f_measure"],
            composite_score=best_score,
            num_trials=stats["num_trials"],
            last_updated=datetime.utcnow().isoformat(),
            confidence=min(1.0, stats["num_trials"] / self.min_trials_for_confidence),
        )

        self._best_configs[suite] = preset
        return preset

    def export_optimized_presets(
        self, output_path: Path | str | None = None
    ) -> dict[str, LearnedPreset]:
        """Export best-performing configurations per suite.

        Args:
            output_path: Optional path to save JSON file

        Returns:
            Dictionary mapping suite names to LearnedPreset
        """
        import sqlite3

        # Get all unique suites from database
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.execute("SELECT DISTINCT suite FROM benchmark_runs")
        suites = [row[0] for row in cursor.fetchall()]
        conn.close()

        presets = {}
        for suite in suites:
            preset = self.get_best_config(suite)
            if preset:
                presets[suite] = preset

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            with open(output_path, "w") as f:
                json.dump(
                    {name: preset.model_dump() for name, preset in presets.items()},
                    f,
                    indent=2,
                )

        return presets

    def import_presets(self, input_path: Path | str) -> None:
        """Import presets from JSON file.

        Args:
            input_path: Path to JSON file with presets
        """
        input_path = Path(input_path)
        with open(input_path) as f:
            data = json.load(f)

        for name, preset_data in data.items():
            self._best_configs[name] = LearnedPreset(**preset_data)

    def _analyze_history(self, suite: str) -> dict[str, Any]:
        """Analyze historical benchmark data for a suite.

        Returns statistics about parameter performance.
        """
        if suite in self._parameter_stats:
            return self._parameter_stats[suite]

        history = self.db.get_history(suite, limit=100)

        if not history:
            result = {"has_data": False}
            self._parameter_stats[suite] = result
            return result

        # Extract parameter-score pairs from stored results
        param_scores = []
        for run in history:
            try:
                results = json.loads(run.results_json)
                config = results.get("config", {})
                if config:
                    param_scores.append(
                        {
                            "params": config,
                            "f_measure": run.avg_f_measure,
                            "precision": run.avg_precision,
                            "recall": run.avg_recall,
                            "mae_ms": run.avg_mae_ms,
                        }
                    )
            except (json.JSONDecodeError, KeyError):
                # Use defaults if config not stored
                param_scores.append(
                    {
                        "params": {"fft_size": 2048, "hop_size": 512},
                        "f_measure": run.avg_f_measure,
                        "precision": run.avg_precision,
                        "recall": run.avg_recall,
                        "mae_ms": run.avg_mae_ms,
                    }
                )

        if not param_scores:
            result = {"has_data": False}
            self._parameter_stats[suite] = result
            return result

        # Find best configuration
        best_idx = np.argmax([p["f_measure"] for p in param_scores])
        best_entry = param_scores[best_idx]

        # Compute statistics per parameter value
        fft_scores = {}
        hop_scores = {}
        threshold_scores = {}

        for entry in param_scores:
            params = entry["params"]
            score = entry["f_measure"]

            fft = params.get("fft_size", 2048)
            hop = params.get("hop_size", 512)
            thresh = params.get("threshold", 0.3)

            if fft not in fft_scores:
                fft_scores[fft] = []
            fft_scores[fft].append(score)

            if hop not in hop_scores:
                hop_scores[hop] = []
            hop_scores[hop].append(score)

            thresh_key = round(thresh, 2)
            if thresh_key not in threshold_scores:
                threshold_scores[thresh_key] = []
            threshold_scores[thresh_key].append(score)

        # Compute average scores per value
        fft_avg = {k: np.mean(v) for k, v in fft_scores.items()}
        hop_avg = {k: np.mean(v) for k, v in hop_scores.items()}
        thresh_avg = {k: np.mean(v) for k, v in threshold_scores.items()}

        # Compute uncertainty (std or based on sample count)
        fft_std = {k: np.std(v) if len(v) > 1 else 0.5 for k, v in fft_scores.items()}
        hop_std = {k: np.std(v) if len(v) > 1 else 0.5 for k, v in hop_scores.items()}

        result = {
            "has_data": True,
            "num_trials": len(param_scores),
            "best_params": best_entry["params"],
            "best_f_measure": best_entry["f_measure"],
            "best_score": best_entry["f_measure"],  # Use f_measure as primary score
            "fft_avg": fft_avg,
            "hop_avg": hop_avg,
            "thresh_avg": thresh_avg,
            "fft_std": fft_std,
            "hop_std": hop_std,
            "all_scores": [p["f_measure"] for p in param_scores],
        }

        self._parameter_stats[suite] = result
        return result

    def _suggest_exploration(
        self, stats: dict[str, Any], exclude: list[dict]
    ) -> SuggestionResult:
        """Suggest parameters for exploration (try uncertain regions)."""
        rng = np.random.default_rng()

        # Find under-explored parameter values
        fft_std = stats.get("fft_std", {})
        hop_std = stats.get("hop_std", {})

        # Parameters with high uncertainty (few trials or high variance)
        # Fallback to standard grid if no history
        possible_fft = [1024, 2048, 4096]
        possible_hop = [256, 512, 1024]
        possible_thresh = [0.2, 0.3, 0.4, 0.5]

        # Prefer under-explored values
        if fft_std:
            # Weight by uncertainty (higher std = less explored)
            weights = [fft_std.get(f, 0.5) + 0.1 for f in possible_fft]
            weights = np.array(weights) / sum(weights)
            fft_size = rng.choice(possible_fft, p=weights)
        else:
            fft_size = rng.choice(possible_fft)

        if hop_std:
            weights = [hop_std.get(h, 0.5) + 0.1 for h in possible_hop]
            weights = np.array(weights) / sum(weights)
            hop_size = rng.choice(possible_hop, p=weights)
        else:
            hop_size = rng.choice(possible_hop)

        threshold = rng.choice(possible_thresh)

        params = {
            "fft_size": int(fft_size),
            "hop_size": int(hop_size),
            "threshold": threshold,
        }

        # Check if already tried
        if params in exclude:
            # Try random variation
            params["fft_size"] = int(rng.choice(possible_fft))
            params["hop_size"] = int(rng.choice(possible_hop))

        return SuggestionResult(
            params=params,
            expected_score=np.mean(stats["all_scores"]) if stats["all_scores"] else 0.5,
            confidence=0.3,
            exploration_weight=0.8,
            reasoning="Exploring under-tested parameter region.",
        )

    def _suggest_exploitation(
        self, stats: dict[str, Any], exclude: list[dict]
    ) -> SuggestionResult:
        """Suggest parameters for exploitation (refine best known)."""
        best = stats["best_params"]

        # Small mutations around best known
        rng = np.random.default_rng()

        fft_options = [1024, 2048, 4096]
        hop_options = [256, 512, 1024]

        current_fft = best.get("fft_size", 2048)
        current_hop = best.get("hop_size", 512)
        current_thresh = best.get("threshold", 0.3)

        # With high probability, keep best values; with low, try neighbor
        if rng.random() > 0.7:
            fft_idx = (
                fft_options.index(current_fft) if current_fft in fft_options else 1
            )
            fft_idx = int(
                np.clip(fft_idx + rng.choice([-1, 0, 1]), 0, len(fft_options) - 1)
            )
            current_fft = fft_options[fft_idx]

        if rng.random() > 0.7:
            hop_idx = (
                hop_options.index(current_hop) if current_hop in hop_options else 1
            )
            hop_idx = int(
                np.clip(hop_idx + rng.choice([-1, 0, 1]), 0, len(hop_options) - 1)
            )
            current_hop = hop_options[hop_idx]

        if rng.random() > 0.7:
            current_thresh = float(
                np.clip(current_thresh + rng.normal(0, 0.05), 0.1, 0.6)
            )
            current_thresh = round(current_thresh, 2)

        params = {
            "fft_size": current_fft,
            "hop_size": current_hop,
            "threshold": current_thresh,
        }

        return SuggestionResult(
            params=params,
            expected_score=stats["best_f_measure"],
            confidence=min(1.0, stats["num_trials"] / self.min_trials_for_confidence),
            exploration_weight=0.2,
            reasoning=f"Refining best known params (F={stats['best_f_measure']:.3f}).",
        )


class PresetManager:
    """Manages learned and static presets together."""

    def __init__(
        self,
        db: BenchmarkDB | None = None,
        learned_presets_path: Path | str | None = None,
    ):
        """Initialize preset manager.

        Args:
            db: Optional BenchmarkDB for learning
            learned_presets_path: Optional path to learned presets file
        """
        self.db = db
        self.tuner = AdaptiveTuner(db) if db else None
        self.learned_presets: dict[str, LearnedPreset] = {}

        if learned_presets_path:
            self.load_learned_presets(learned_presets_path)

    def get_preset(self, name: str) -> AnalyzerConfig:
        """Get analyzer config by preset name.

        First checks learned presets, then falls back to static presets.

        Args:
            name: Preset name

        Returns:
            AnalyzerConfig
        """
        # Check learned presets
        if name in self.learned_presets:
            preset = self.learned_presets[name]
            return AnalyzerConfig(
                fft_size=preset.config.get("fft_size", 2048),
                hop_size=preset.config.get("hop_size", 512),
                sample_rate=preset.config.get("sample_rate", 44100),
            )

        # Check static presets
        from aubio_beatcheck.presets import PRESETS

        if name in PRESETS:
            return PRESETS[name]

        raise KeyError(f"Unknown preset: {name}")

    def update_learned_preset(
        self,
        suite: str,
        config: AnalyzerConfig,
        f_measure: float,
        composite_score: float,
    ) -> None:
        """Update or create learned preset.

        Args:
            suite: Suite name (becomes preset name)
            config: Analyzer configuration
            f_measure: Achieved F-measure
            composite_score: Composite optimization score
        """
        existing = self.learned_presets.get(suite)

        # Only update if better
        if existing and existing.composite_score >= composite_score:
            return

        self.learned_presets[suite] = LearnedPreset(
            name=suite,
            suite=suite,
            config={
                "fft_size": config.fft_size,
                "hop_size": config.hop_size,
                "sample_rate": config.sample_rate,
            },
            f_measure=f_measure,
            composite_score=composite_score,
            num_trials=(existing.num_trials + 1) if existing else 1,
            last_updated=datetime.utcnow().isoformat(),
            confidence=min(1.0, ((existing.num_trials if existing else 0) + 1) / 10),
        )

    def save_learned_presets(self, path: Path | str) -> None:
        """Save learned presets to file.

        Args:
            path: Output file path
        """
        path = Path(path)
        with open(path, "w") as f:
            json.dump(
                {
                    name: preset.model_dump()
                    for name, preset in self.learned_presets.items()
                },
                f,
                indent=2,
            )

    def load_learned_presets(self, path: Path | str) -> None:
        """Load learned presets from file.

        Args:
            path: Input file path
        """
        path = Path(path)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            self.learned_presets = {
                name: LearnedPreset(**preset_data) for name, preset_data in data.items()
            }

    def list_all_presets(self) -> dict[str, dict[str, Any]]:
        """List all available presets with metadata.

        Returns:
            Dictionary mapping preset names to info
        """
        from aubio_beatcheck.presets import PRESETS

        result = {}

        # Static presets
        for name, config in PRESETS.items():
            result[name] = {
                "type": "static",
                "fft_size": config.fft_size,
                "hop_size": config.hop_size,
                "sample_rate": config.sample_rate,
            }

        # Learned presets
        for name, preset in self.learned_presets.items():
            result[f"learned_{name}"] = {
                "type": "learned",
                "fft_size": preset.config["fft_size"],
                "hop_size": preset.config["hop_size"],
                "sample_rate": preset.config["sample_rate"],
                "f_measure": preset.f_measure,
                "confidence": preset.confidence,
            }

        return result
