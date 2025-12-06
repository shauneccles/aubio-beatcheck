"""Core analysis and signal generation functionality."""

from .adaptive_tuner import AdaptiveTuner, LearnedPreset, PresetManager
from .analyzer_factory import (
    AnalyzerFactory,
    AnalyzerSelector,
    SignalClassifier,
    SignalProfile,
)
from .analyzers import OnsetAnalyzer, PitchAnalyzer, PvocAnalyzer, TempoAnalyzer
from .feedback_loop import FeedbackAction, FeedbackLoop, PerformanceMonitor, TrendReport
from .multi_objective import (
    CompositeScore,
    MultiObjectiveEvaluator,
    ObjectivePresets,
    OptimizationObjectives,
    ParetoOptimizer,
)
from .signals import SignalGenerator

__all__ = [
    # Analyzers
    "TempoAnalyzer",
    "OnsetAnalyzer",
    "PitchAnalyzer",
    "PvocAnalyzer",
    # Signal Generation
    "SignalGenerator",
    # Multi-Objective Optimization
    "OptimizationObjectives",
    "CompositeScore",
    "MultiObjectiveEvaluator",
    "ParetoOptimizer",
    "ObjectivePresets",
    # Feedback Loop
    "FeedbackAction",
    "FeedbackLoop",
    "PerformanceMonitor",
    "TrendReport",
    # Analyzer Factory
    "SignalProfile",
    "SignalClassifier",
    "AnalyzerFactory",
    "AnalyzerSelector",
    # Adaptive Tuning
    "AdaptiveTuner",
    "LearnedPreset",
    "PresetManager",
]
