"""Benchmark Comparison Utilities.

Compare benchmark results against baselines and generate reports.
"""

from typing import Any

from pydantic import BaseModel, Field


class SignalComparison(BaseModel):
    """Comparison of a single signal's results."""

    signal_name: str
    baseline_f_measure: float | None = None
    current_f_measure: float | None = None
    f_measure_delta: float = 0.0
    improved: bool = False
    regressed: bool = False
    baseline_mae_ms: float | None = None
    current_mae_ms: float | None = None


class ComparisonReport(BaseModel):
    """Comprehensive comparison between two benchmark runs."""

    summary: str = Field(description="Human-readable summary")
    improved_count: int = Field(description="Number of improved signals")
    regressed_count: int = Field(description="Number of regressed signals")
    unchanged_count: int = Field(description="Number of unchanged signals")
    overall_improved: bool = Field(description="True if overall metrics improved")
    overall_regressed: bool = Field(description="True if overall metrics regressed")
    avg_f_measure_baseline: float = Field(description="Baseline average F-measure")
    avg_f_measure_current: float = Field(description="Current average F-measure")
    avg_f_measure_delta: float = Field(description="Change in average F-measure")
    signal_comparisons: list[SignalComparison] = Field(
        default_factory=list, description="Per-signal comparisons"
    )


def compare_results(
    baseline: dict[str, Any],
    current: dict[str, Any],
    threshold: float = 0.02,
) -> ComparisonReport:
    """Compare two sets of benchmark results.

    Args:
        baseline: Baseline results dictionary.
        current: Current results dictionary.
        threshold: Threshold for detecting significant changes.

    Returns:
        ComparisonReport with detailed comparison.
    """
    baseline_signals = {s["signal_name"]: s for s in baseline.get("signals", [])}
    current_signals = {s["signal_name"]: s for s in current.get("signals", [])}

    signal_comparisons: list[SignalComparison] = []
    improved = 0
    regressed = 0
    unchanged = 0

    baseline_f_measures = []
    current_f_measures = []

    all_names = set(baseline_signals.keys()) | set(current_signals.keys())

    for name in sorted(all_names):
        base = baseline_signals.get(name)
        curr = current_signals.get(name)

        base_f = None
        curr_f = None
        base_mae = None
        curr_mae = None

        if base and base.get("evaluation"):
            base_f = base["evaluation"].get("f_measure", 0)
            base_mae = base["evaluation"].get("mean_absolute_error_ms", 0)
            baseline_f_measures.append(base_f)

        if curr and curr.get("evaluation"):
            curr_f = curr["evaluation"].get("f_measure", 0)
            curr_mae = curr["evaluation"].get("mean_absolute_error_ms", 0)
            current_f_measures.append(curr_f)

        delta = 0.0
        is_improved = False
        is_regressed = False

        if base_f is not None and curr_f is not None:
            delta = curr_f - base_f
            if delta > threshold:
                is_improved = True
                improved += 1
            elif delta < -threshold:
                is_regressed = True
                regressed += 1
            else:
                unchanged += 1

        signal_comparisons.append(
            SignalComparison(
                signal_name=name,
                baseline_f_measure=base_f,
                current_f_measure=curr_f,
                f_measure_delta=delta,
                improved=is_improved,
                regressed=is_regressed,
                baseline_mae_ms=base_mae,
                current_mae_ms=curr_mae,
            )
        )

    avg_base = (
        sum(baseline_f_measures) / len(baseline_f_measures)
        if baseline_f_measures
        else 0
    )
    avg_curr = (
        sum(current_f_measures) / len(current_f_measures) if current_f_measures else 0
    )
    avg_delta = avg_curr - avg_base

    overall_improved = avg_delta > threshold
    overall_regressed = avg_delta < -threshold

    if overall_improved:
        summary = f"✅ IMPROVED: Average F-measure {avg_base:.3f} → {avg_curr:.3f} (+{avg_delta:.3f})"
    elif overall_regressed:
        summary = f"❌ REGRESSED: Average F-measure {avg_base:.3f} → {avg_curr:.3f} ({avg_delta:.3f})"
    else:
        summary = f"➖ UNCHANGED: Average F-measure {avg_base:.3f} → {avg_curr:.3f} ({avg_delta:+.3f})"

    return ComparisonReport(
        summary=summary,
        improved_count=improved,
        regressed_count=regressed,
        unchanged_count=unchanged,
        overall_improved=overall_improved,
        overall_regressed=overall_regressed,
        avg_f_measure_baseline=avg_base,
        avg_f_measure_current=avg_curr,
        avg_f_measure_delta=avg_delta,
        signal_comparisons=signal_comparisons,
    )


def format_comparison_table(report: ComparisonReport) -> str:
    """Format comparison report as a markdown table.

    Args:
        report: ComparisonReport to format.

    Returns:
        Markdown-formatted table string.
    """
    lines = [
        "| Signal | Baseline F1 | Current F1 | Delta | Status |",
        "|--------|-------------|------------|-------|--------|",
    ]

    for s in report.signal_comparisons:
        base = (
            f"{s.baseline_f_measure:.3f}" if s.baseline_f_measure is not None else "N/A"
        )
        curr = (
            f"{s.current_f_measure:.3f}" if s.current_f_measure is not None else "N/A"
        )
        delta = f"{s.f_measure_delta:+.3f}"

        if s.improved:
            status = "✅ Improved"
        elif s.regressed:
            status = "❌ Regressed"
        else:
            status = "➖ Unchanged"

        lines.append(f"| {s.signal_name} | {base} | {curr} | {delta} | {status} |")

    lines.append("")
    lines.append(f"**Summary:** {report.summary}")
    lines.append(f"- Improved: {report.improved_count}")
    lines.append(f"- Regressed: {report.regressed_count}")
    lines.append(f"- Unchanged: {report.unchanged_count}")

    return "\n".join(lines)
