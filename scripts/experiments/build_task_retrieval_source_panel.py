from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from token_prediction.canonical_json import canonical_json_bytes
from token_prediction.evaluation.multiple_comparisons import holm_adjusted
from token_prediction.experiments.retrieval_analysis import (
    _LINEAR_METRICS,
    _candidate_summary,
    _comparison_pairs,
    _paired_comparison,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROW_METRICS = (*_LINEAR_METRICS, "absolute_error_mass", "absolute_target_mass")
EXPERIMENTS = ("in_call_update", "task_update")
PRIMARY_CANDIDATES = {
    "in_call_update": ("ic_selected", "ic_task_knn"),
    "task_update": ("tu_selected", "tu_task_knn"),
}
SOURCE_INPUT_FILES = (
    "analysis-plan.json",
    "grid.json",
    "manifest.json",
    "task-retrieval-analysis.json",
    "task-retrieval-grid-audit.json",
    "task-retrieval-replay-audit.json",
    "task-retrieval-pooled-tasks.jsonl",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_source(
    source_key: str,
    root: Path,
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"source root is not a directory: {root}")
    paths = {name: root / name for name in SOURCE_INPUT_FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"source {source_key} lacks required files: {missing}")

    source_plan = _load_json(paths["analysis-plan.json"])
    if source_plan != dict(plan):
        raise ValueError(f"source {source_key} uses a different analysis plan")
    grid = _load_json(paths["grid.json"])
    if grid.get("config_id") != plan.get("config_id"):
        raise ValueError(f"source {source_key} grid config_id differs from the plan")
    if grid.get("split_seeds") != plan.get("split_seeds"):
        raise ValueError(f"source {source_key} split seeds differ from the plan")
    if grid.get("fit_seeds") != plan["fit_seed_policy"]["formal_fit_seeds"]:
        raise ValueError(f"source {source_key} fit seeds differ from the plan")

    analysis = _load_json(paths["task-retrieval-analysis.json"])
    if analysis.get("evaluation_scope") != "development_only":
        raise ValueError(f"source {source_key} is outside development scope")
    if analysis.get("final_holdout_status") != "unassigned":
        raise ValueError(f"source {source_key} changed final-holdout status")
    if _load_json(paths["task-retrieval-grid-audit.json"]).get("status") != "passed":
        raise ValueError(f"source {source_key} grid audit is not passed")
    if _load_json(paths["task-retrieval-replay-audit.json"]).get("status") != "passed":
        raise ValueError(f"source {source_key} replay audit is not passed")

    rows: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    with paths["task-retrieval-pooled-tasks.jsonl"].open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"source {source_key} row {line_number} is not an object")
            experiment = raw.get("experiment")
            candidate = raw.get("candidate_id")
            task_id = raw.get("task_id")
            if experiment not in EXPERIMENTS or not isinstance(candidate, str) or not candidate:
                raise ValueError(f"source {source_key} row {line_number} has invalid identity")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"source {source_key} row {line_number} has invalid task_id")
            candidate_rows = rows.setdefault(experiment, {}).setdefault(candidate, {})
            if task_id in candidate_rows:
                raise ValueError(f"source {source_key} has duplicate pooled Task rows")
            candidate_rows[task_id] = {
                name: _finite_number(raw.get(name), name=f"{source_key}:{name}")
                for name in SOURCE_ROW_METRICS
            }

    if set(rows) != set(EXPERIMENTS):
        raise ValueError(f"source {source_key} does not contain both experiment cells")
    candidate_sets = {experiment: set(rows[experiment]) for experiment in EXPERIMENTS}
    if not candidate_sets["in_call_update"] or not candidate_sets["task_update"]:
        raise ValueError(f"source {source_key} has an empty candidate cell")
    for experiment, candidates in rows.items():
        rosters = {frozenset(candidate_rows) for candidate_rows in candidates.values()}
        if len(rosters) != 1:
            raise ValueError(f"source {source_key} has inconsistent Task rosters")
        if len(next(iter(rosters))) != 69:
            raise ValueError(f"source {source_key} expected 69 Tasks in {experiment}")

    source_manifest = _load_json(paths["manifest.json"])
    artifact_id = source_manifest.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError(f"source {source_key} manifest lacks artifact_id")
    return {
        "source_key": source_key,
        "root": _relative_path(root),
        "artifact_id": artifact_id,
        "grid_id": grid.get("grid_id"),
        "rows": rows,
        "input_files": {
            name: {
                "path": _relative_path(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
    }


def _panel_tasks(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    source_keys = sorted(sources)
    if not source_keys:
        raise ValueError("source panel requires at least one source")
    panel: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for experiment in EXPERIMENTS:
        candidate_sets = [set(sources[key]["rows"][experiment]) for key in source_keys]
        if any(candidate_sets[0] != candidates for candidates in candidate_sets[1:]):
            raise ValueError(f"source panel candidate sets differ in {experiment}")
        panel[experiment] = {}
        for candidate in sorted(candidate_sets[0]):
            task_sets = [set(sources[key]["rows"][experiment][candidate]) for key in source_keys]
            if any(task_sets[0] != tasks for tasks in task_sets[1:]):
                raise ValueError(f"source panel Task rosters differ for {candidate}")
            panel[experiment][candidate] = {
                task_id: {
                    name: sum(
                        sources[key]["rows"][experiment][candidate][task_id][name]
                        for key in source_keys
                    )
                    / len(source_keys)
                    for name in SOURCE_ROW_METRICS
                }
                for task_id in sorted(task_sets[0])
            }
    return panel


def _source_summaries(
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    return {
        source_key: {
            experiment: {
                candidate: _candidate_summary(source["rows"][experiment][candidate])
                for candidate in sorted(source["rows"][experiment])
            }
            for experiment in EXPERIMENTS
        }
        for source_key, source in sorted(sources.items())
    }


def _experiment_for_pair(
    panel: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    candidate: str,
    reference: str,
) -> str:
    matches = [
        experiment
        for experiment in EXPERIMENTS
        if candidate in panel[experiment] and reference in panel[experiment]
    ]
    if len(matches) != 1:
        raise ValueError(f"comparison {candidate}/{reference} does not identify one experiment")
    return matches[0]


def _comparisons(
    plan: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
    panel: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    comparisons: dict[str, Any] = {}
    corrections: dict[str, Any] = {}
    bootstrap = plan["bootstrap"]
    multiplicity = plan["multiplicity"]
    for family_name in ("primary", "mechanism", "sensitivity"):
        family: dict[str, Any] = {}
        raw_p: dict[str, float] = {}
        for pair_index, (candidate, reference) in enumerate(
            _comparison_pairs(plan, family_name)
        ):
            experiment = _experiment_for_pair(panel, candidate, reference)
            if family_name == "primary":
                randomization_seed = int(
                    multiplicity["primary_randomization_seeds"][experiment]
                )
            else:
                randomization_seed = int(
                    multiplicity[f"{family_name}_randomization_seed_base"]
                ) + pair_index
            comparison = _paired_comparison(
                panel[experiment][candidate],
                panel[experiment][reference],
                bootstrap_iterations=int(bootstrap["iterations"]),
                bootstrap_seed=int(bootstrap["seeds"][experiment]),
                randomization_iterations=int(multiplicity["primary_randomization_iterations"]),
                randomization_seed=randomization_seed,
            )
            comparison["by_source"] = {
                source_key: _paired_comparison(
                    source["rows"][experiment][candidate],
                    source["rows"][experiment][reference],
                    bootstrap_iterations=int(bootstrap["iterations"]),
                    bootstrap_seed=int(bootstrap["seeds"][experiment]),
                    randomization_iterations=None,
                    randomization_seed=randomization_seed,
                    require_relative=False,
                )
                for source_key, source in sorted(sources.items())
            }
            comparison.update(
                {
                    "experiment": experiment,
                    "candidate_id": candidate,
                    "reference_id": reference,
                }
            )
            label = f"{candidate}__vs__{reference}"
            family[label] = comparison
            raw_p[label] = comparison["sign_flip_p"]
        adjusted = holm_adjusted(raw_p) if raw_p else {}
        for label, value in adjusted.items():
            family[label]["holm_adjusted_p"] = value
            if family_name == "primary":
                family[label]["decision"] = {
                    "classification": "descriptive_partial_source_panel",
                    "claim_scope": "task_retrieval_component_development_descriptive",
                    "eligible_for_formal_decision": False,
                    "observed_source_keys": sorted(sources),
                    "required_source_keys": sorted(plan["validation"]["source_keys"]),
                }
        comparisons[family_name] = family
        corrections[family_name] = {
            "family_size": len(family),
            "raw_p": raw_p,
            "holm_adjusted_p": adjusted,
        }
    return comparisons, corrections


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row)) + b"\n")


def _file_identity(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _summary_markdown(
    panel_document: Mapping[str, Any],
    *,
    output_path: Path,
) -> str:
    lines = [
        "# Task-retrieval v18/v21 source-subset panel",
        "",
        "This deterministic audit averages the frozen v18 and v21 source conditions equally within each shared Task.",
        "It performs no fitting and makes no provider calls.",
        "",
        "- Status: descriptive partial source panel",
        "- Sources: `v18`, `v21`",
        "- Retired source: `v19` after the recorded user-directed stop",
        "- Unique Tasks: 69 per experiment",
        "- Evaluation scope: development only",
        "- Final holdout: unassigned",
        f"- Output: `{_relative_path(output_path)}`",
        "",
        "| Experiment | Candidate | Reference | Candidate MAE | Reference MAE | Relative delta | 95% CI | Sign-flip p |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for experiment in EXPERIMENTS:
        candidate, reference = PRIMARY_CANDIDATES[experiment][1], PRIMARY_CANDIDATES[experiment][0]
        comparison = panel_document["comparisons"]["primary"][f"{candidate}__vs__{reference}"]
        mae = comparison["metrics"]["mae"]
        lines.append(
            f"| `{experiment}` | `{candidate}` | `{reference}` | "
            f"{mae['candidate']:.6f} | {mae['reference']:.6f} | "
            f"{mae['relative_delta'] * 100:.3f}% | "
            f"[{mae['relative_ci_lower'] * 100:.3f}%, {mae['relative_ci_upper'] * 100:.3f}%] | "
            f"{comparison['sign_flip_p']:.6g} |"
        )
    lines.extend(
        [
            "",
            "The panel preserves source-specific comparisons and all candidate arms in `source-panel.json`.",
            "The frozen three-source analysis plan remains the authority for formal promotion; this source subset carries descriptive evidence only.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_source_panel(
    source_roots: Mapping[str, str | Path],
    *,
    output_dir: str | Path,
    analysis_plan_path: str | Path,
) -> Path:
    if set(source_roots) != {"v18", "v21"}:
        raise ValueError("the source-subset panel requires exactly v18 and v21")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    plan = _load_json(Path(analysis_plan_path).resolve())
    sources = {
        source_key: _load_source(source_key, Path(root), plan=plan)
        for source_key, root in sorted(source_roots.items())
    }
    panel_tasks = _panel_tasks(sources)
    comparisons, corrections = _comparisons(plan, sources, panel_tasks)
    source_keys = sorted(sources)
    task_ids = sorted(set(panel_tasks["in_call_update"]["ic_selected"]))
    if task_ids != sorted(set(panel_tasks["task_update"]["tu_selected"])):
        raise ValueError("the two experiment cells use different shared Task rosters")
    panel_document = {
        "schema_id": "task-retrieval-source-panel-v1",
        "panel_id": "task-retrieval-v18-v21-source-subset",
        "analysis_id": plan["analysis_id"],
        "evaluation_scope": "development_only",
        "final_holdout_status": "unassigned",
        "source_panel": {
            "status": "partial",
            "eligible_for_formal_decision": False,
            "observed_source_keys": source_keys,
            "required_source_keys": sorted(plan["validation"]["source_keys"]),
            "excluded_source_keys": ["v19"],
            "exclusion_reason": "user-directed stop recorded in the v19 condition-retirement report",
        },
        "aggregation": {
            "source_condition_weight": "equal",
            "task_weight": "equal",
            "order": "source-condition-mean-within-task-then-unique-task-mean",
            "paired_unit": "unique_task_id",
            "source_rows": "frozen task-retrieval-pooled-tasks.jsonl",
        },
        "split_seeds": plan["split_seeds"],
        "fit_seeds": plan["fit_seed_policy"]["formal_fit_seeds"],
        "n_tasks": len(task_ids),
        "source_summaries": _source_summaries(sources),
        "candidate_summaries": {
            experiment: {
                candidate: _candidate_summary(panel_tasks[experiment][candidate])
                for candidate in sorted(panel_tasks[experiment])
            }
            for experiment in EXPERIMENTS
        },
        "comparisons": comparisons,
        "family_corrections": corrections,
        "sources": {
            source_key: {
                key: value
                for key, value in source.items()
                if key != "rows"
            }
            for source_key, source in sorted(sources.items())
        },
    }
    source_rows = []
    for source_key, source in sorted(sources.items()):
        for experiment in EXPERIMENTS:
            for candidate in sorted(source["rows"][experiment]):
                for task_id, values in sorted(source["rows"][experiment][candidate].items()):
                    source_rows.append(
                        {
                            "level": "source_task",
                            "source_key": source_key,
                            "experiment": experiment,
                            "candidate_id": candidate,
                            "task_id": task_id,
                            **values,
                        }
                    )
    panel_rows = []
    for experiment in EXPERIMENTS:
        for candidate in sorted(panel_tasks[experiment]):
            for task_id, values in sorted(panel_tasks[experiment][candidate].items()):
                panel_rows.append(
                    {
                        "level": "panel_task",
                        "experiment": experiment,
                        "candidate_id": candidate,
                        "task_id": task_id,
                        **values,
                    }
                )
    _write_jsonl(output / "source-panel-tasks.jsonl", (*source_rows, *panel_rows))
    _write_json(output / "source-panel.json", panel_document)
    summary = _summary_markdown(panel_document, output_path=output / "source-panel.json")
    (output / "summary.md").write_text(summary, encoding="utf-8")
    manifest = {
        "schema_id": "task-retrieval-source-panel-manifest-v1",
        "artifact_id": "task-retrieval-v18-v21-source-subset",
        "stage_name": "task-retrieval-source-panel",
        "source_keys": source_keys,
        "source_artifacts": panel_document["sources"],
        "files": {
            name: _file_identity(output / name)
            for name in ("source-panel.json", "source-panel-tasks.jsonl", "summary.md")
        },
    }
    _write_json(output / "manifest.json", manifest)
    return output


def _source_roots(values: Sequence[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        source_key, separator, raw_path = value.partition("=")
        if not separator or source_key in roots or not raw_path:
            raise ValueError("sources must use unique SOURCE_KEY=GRID_PATH arguments")
        roots[source_key] = Path(raw_path)
    return roots


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--analysis-plan", required=True, type=Path)
    arguments = parser.parse_args(argv)
    result = build_source_panel(
        _source_roots(arguments.source),
        output_dir=arguments.output,
        analysis_plan_path=arguments.analysis_plan,
    )
    print(result)


if __name__ == "__main__":
    main()
