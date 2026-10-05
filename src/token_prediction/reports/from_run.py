from __future__ import annotations

from typing import Any, Mapping, Sequence


DEFAULT_GRID_METRICS = ("wape", "coverage", "interval_score")


def _series_cell(series: Mapping[str, Any]) -> str:
    mean = series.get("mean")
    std = series.get("std")
    if mean is None:
        return "-"
    if std is None:
        return f"{mean:.4g}"
    return f"{mean:.4g} ± {std:.2g}"


def format_seed_grid_markdown(
    document: Mapping[str, Any],
    *,
    metrics: Sequence[str] = DEFAULT_GRID_METRICS,
) -> str:


    aggregates = document.get("aggregates")
    if not isinstance(aggregates, Mapping) or not aggregates:
        raise ValueError("seed grid document has no aggregates")
    header = "| experiment | candidate | " + " | ".join(metrics) + " |"
    rule = "| --- | --- | " + " | ".join("---:" for _ in metrics) + " |"
    lines = [header, rule]
    for experiment_id in sorted(aggregates):
        candidates = aggregates[experiment_id]
        for candidate_id in sorted(candidates):
            bucket = candidates[candidate_id]
            cells = " | ".join(
                _series_cell(bucket.get(metric, {})) for metric in metrics
            )
            lines.append(f"| {experiment_id} | {candidate_id} | {cells} |")
    lines.append("")
    lines.append(
        f"Fit seeds {document.get('fit_seeds')} x split seeds "
        f"{document.get('split_seeds')}; grid {document.get('grid_id')}."
    )
    return "\n".join(lines)


def format_learning_curve_markdown(
    document: Mapping[str, Any],
    *,
    metrics: Sequence[str] = DEFAULT_GRID_METRICS,
) -> str:


    aggregates = document.get("aggregates")
    if not isinstance(aggregates, Mapping) or not aggregates:
        raise ValueError("learning curve document has no aggregates")
    counts = {
        point["fraction"]: point.get("train_task_count")
        for point in document.get("points", [])
    }
    lines: list[str] = []
    for experiment_id in sorted(aggregates):
        candidates = aggregates[experiment_id]
        for candidate_id in sorted(candidates):
            bucket = candidates[candidate_id]
            fractions = sorted(
                {
                    entry["fraction"]
                    for metric in metrics
                    for entry in bucket.get(metric, [])
                }
            )
            if not fractions:
                continue
            lines.append(f"### {experiment_id} / {candidate_id}")
            lines.append(
                "| fraction | mean train tasks | " + " | ".join(metrics) + " |"
            )
            lines.append(
                "| ---: | ---: | " + " | ".join("---:" for _ in metrics) + " |"
            )
            for fraction in fractions:
                cells = []
                for metric in metrics:
                    entry = next(
                        (
                            item
                            for item in bucket.get(metric, [])
                            if item["fraction"] == fraction
                        ),
                        None,
                    )
                    cells.append(_series_cell(entry) if entry else "-")
                count = counts.get(fraction)
                count_text = f"{count:.1f}" if isinstance(count, (int, float)) else "-"
                lines.append(
                    f"| {fraction} | {count_text} | " + " | ".join(cells) + " |"
                )
            lines.append("")
    if not lines:
        raise ValueError("learning curve document has no plottable series")
    lines.append(f"Curve {document.get('curve_id')}.")
    return "\n".join(lines)


def format_sweep_markdown(
    document: Mapping[str, Any],
    *,
    metrics: Sequence[str] = DEFAULT_GRID_METRICS,
) -> str:


    if not document:


        return "No hyperparameter sweep was declared for this run."
    lines: list[str] = []
    for sweep_id in sorted(document):
        sweep = document[sweep_id]
        if not isinstance(sweep, Mapping):
            raise ValueError(f"sweep {sweep_id!r} is not a sweep document")
        cells = sweep.get("cells", {})
        for cell_id in sorted(cells):
            points = cells[cell_id]
            lines.append(f"### {sweep_id} — {sweep.get('param')} — {cell_id}")
            lines.append("| value | candidate | " + " | ".join(metrics) + " |")
            lines.append(
                "| --- | --- | " + " | ".join("---:" for _ in metrics) + " |"
            )
            for point in points:
                row_metrics = point.get("metrics", {})
                formatted = []
                for metric in metrics:
                    value = row_metrics.get(metric)
                    if isinstance(value, Mapping):
                        formatted.append(_series_cell(value))
                    elif isinstance(value, (int, float)):
                        formatted.append(f"{value:.4g}")
                    else:
                        formatted.append("-")
                lines.append(
                    f"| {point.get('value')} | {point.get('candidate_id')} | "
                    + " | ".join(formatted)
                    + " |"
                )
            lines.append("")
    if not lines:
        raise ValueError("sweep document has no curves")
    return "\n".join(lines)


def format_pareto_markdown(document: Mapping[str, Any]) -> str:


    rows = document.get("rows")
    if not rows:
        raise ValueError("pareto document has no rows")
    lines = [
        "| budget | policy | decision input | spent tokens | success rate | "
        "wrongly stopped rate |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['budget']} | {row['policy_id']} | {row['decision_input']} | "
            f"{row['spent_tokens']} | {row['success_rate']:.3f} | "
            f"{row['wrongly_stopped_rate']:.3f} |"
        )
    lines.append("")
    lines.append(
        f"{document.get('n_runs')} replayed runs; policy suite "
        f"{document.get('policy_suite_id')}."
    )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_GRID_METRICS",
    "format_learning_curve_markdown",
    "format_pareto_markdown",
    "format_seed_grid_markdown",
    "format_sweep_markdown",
]
