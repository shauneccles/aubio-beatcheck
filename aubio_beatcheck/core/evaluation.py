"""
Evaluation module for comparing detected events against ground truth.
"""

import numpy as np
from typing import List, Tuple
from pydantic import BaseModel, Field


class EvaluationMetrics(BaseModel):
    """Metrics for evaluating detection accuracy."""

    precision: float = Field(default=0.0, ge=0, le=1, description="Precision score")
    recall: float = Field(default=0.0, ge=0, le=1, description="Recall score")
    f_measure: float = Field(default=0.0, ge=0, le=1, description="F1 score")
    mean_absolute_error_ms: float = Field(
        default=0.0, ge=0, description="Mean timing error in ms"
    )
    false_positives: List[float] = Field(
        default_factory=list, description="Timestamps of false positives"
    )
    false_negatives: List[float] = Field(
        default_factory=list, description="Timestamps of missed events"
    )
    matched_events: List[Tuple[float, float]] = Field(
        default_factory=list, description="(ground_truth, detected) pairs"
    )


class Evaluator:
    """Evaluates detected events against ground truth."""

    @staticmethod
    def evaluate_events(
        detected: List[float], ground_truth: List[float], tolerance_ms: float = 50.0
    ) -> EvaluationMetrics:
        """
        Evaluate detected events (beats/onsets) against ground truth.

        Args:
            detected: List of detected timestamps (seconds)
            ground_truth: List of ground truth timestamps (seconds)
            tolerance_ms: Tolerance window in milliseconds

        Returns:
            EvaluationMetrics object
        """
        tolerance_sec = tolerance_ms / 1000.0

        detected = np.array(detected)
        ground_truth = np.array(ground_truth)

        if len(ground_truth) == 0:
            return EvaluationMetrics(
                precision=0.0 if len(detected) > 0 else 1.0,
                recall=1.0,
                f_measure=0.0 if len(detected) > 0 else 1.0,
                false_positives=detected.tolist(),
            )

        if len(detected) == 0:
            return EvaluationMetrics(
                precision=1.0,
                recall=0.0,
                f_measure=0.0,
                false_negatives=ground_truth.tolist(),
            )

        # Simple greedy matching
        matched_gt_indices = set()
        matched_det_indices = set()
        matches = []
        errors = []

        for i, gt_time in enumerate(ground_truth):
            distances = np.abs(detected - gt_time)
            min_dist_idx = np.argmin(distances)
            min_dist = distances[min_dist_idx]

            if min_dist <= tolerance_sec:
                if min_dist_idx not in matched_det_indices:
                    matched_gt_indices.add(i)
                    matched_det_indices.add(min_dist_idx)
                    matches.append((float(gt_time), float(detected[min_dist_idx])))
                    errors.append(min_dist)

        # Calculate metrics
        tp = len(matches)
        fp_indices = set(range(len(detected))) - matched_det_indices
        fn_indices = set(range(len(ground_truth))) - matched_gt_indices

        fp = len(fp_indices)
        fn = len(fn_indices)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_measure = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        mae_ms = (np.mean(errors) * 1000.0) if errors else 0.0

        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f_measure=f_measure,
            mean_absolute_error_ms=mae_ms,
            false_positives=[float(detected[i]) for i in sorted(fp_indices)],
            false_negatives=[float(ground_truth[i]) for i in sorted(fn_indices)],
            matched_events=matches,
        )
