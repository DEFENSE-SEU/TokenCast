from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import time

import numpy as np
import lightgbm as lgb
import requests
import tiktoken

from .predict_envbench_four_actor_once import ROW_FIELDS
from .score_envbench_four_actor_once import (
    ARMS, exact_binomial_upper, git_blob_oid, nonnegative_int, parse_source,
    result_task_id, score_case, verify_predictions,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/envbench_python_transfer_20260923"
AUDIT = ROOT / "workspace/experiments/envbench_python_second_source_20260923/source_feasibility.json"
JVM_PROTOCOL = ROOT / "workspace/experiments/envbench_four_actor_20260923/protocol.lock.json"
ACTOR = "python_bash_agent-4o"
PARSER_ALIAS = "jvm_bash_agent-4o"


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exclusive_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")


def preflight() -> tuple[dict, dict, dict[str, list[dict]], str]:
    lock_path = BASE / "protocol.lock.json"
    receipt_path = BASE / "prediction_receipt.json"
    review_path = BASE / "pretest_review_receipt.json"
    predictions_path = BASE / "predictions.jsonl"
    if not all(p.exists() for p in (lock_path, receipt_path, review_path, predictions_path,
                                    BASE / "prediction_started.json")):
        raise ValueError("Protocol, review, predictions, or prediction receipt absent")
    if (BASE / "score_started.json").exists():
        raise FileExistsError("One-shot score already started")
    lock_sha = digest(lock_path)
    protocol = json.loads(lock_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    pred_script = ROOT / "scripts/experiments/predict_envbench_python_transfer_once.py"
    if (protocol.get("status") != "zero_python_tuning_transfer_locked"
            or protocol.get("source_audit_sha256") != digest(AUDIT)
            or protocol.get("jvm_protocol_sha256") != digest(JVM_PROTOCOL)
            or protocol.get("code_sha256", {}).get("predictor") != digest(pred_script)
            or protocol.get("code_sha256", {}).get("scorer") != digest(Path(__file__))
            or protocol.get("actor") != ACTOR
            or protocol.get("tokenizer") != "o200k_base"
            or protocol.get("tiktoken_version") != tiktoken.__version__
            or protocol.get("numpy_version") != np.__version__
            or protocol.get("lightgbm_version") != lgb.__version__
            or protocol.get("policy") != {"cap": 47078, "fit_offset": 9,
                                           "lower_offset": -64, "gate": "one_tool_no_reset"}):
        raise ValueError("Scorer source or locked protocol changed")
    jvm = json.loads(JVM_PROTOCOL.read_text(encoding="utf-8"))
    imported = {
        "predictor": digest(ROOT / "scripts/experiments/predict_envbench_four_actor_once.py"),
        "scorer": digest(ROOT / "scripts/experiments/score_envbench_four_actor_once.py"),
    }
    if (imported != protocol.get("imported_jvm_code_sha256")
            or imported != {key: jvm.get("code_sha256", {}).get(key)
                            for key in ("predictor", "scorer")}):
        raise ValueError("Imported JVM prediction/scoring code changed")
    feature_code = ROOT / "scripts/experiments/develop_envbench_jvm_budget_models.py"
    if protocol.get("transitive_feature_code_sha256") != digest(feature_code):
        raise ValueError("Imported JVM feature-vector code changed")
    if (review.get("status") != "pass" or review.get("protocol_sha256") != lock_sha
            or review.get("predictor_sha256") != digest(pred_script)
            or review.get("scorer_sha256") != digest(Path(__file__))):
        raise ValueError("Independent pretest review receipt changed")
    if (receipt.get("status") != "predictions_locked_before_any_nonprobe_result_parse"
            or receipt.get("protocol_sha256") != lock_sha
            or receipt.get("predictions_sha256") != digest(predictions_path)
            or receipt.get("nonprobe_result_opened") is not False
            or receipt.get("completed_tasks") != 327):
        raise ValueError("Prediction receipt incomplete or changed")
    ids = protocol.get("test_ids")
    if not isinstance(ids, list) or len(ids) != 327 or len(set(ids)) != 327:
        raise ValueError("Test identity roster invalid")
    if set(receipt.get("source_sha256", {})) != set(ids) or set(receipt.get("row_count_by_task", {})) != set(ids):
        raise ValueError("Source/prediction coverage incomplete")
    predictions: dict[str, list[dict]] = defaultdict(list)
    with predictions_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (set(row) != set(ROW_FIELDS) or row.get("actor") != ACTOR
                    or row.get("task_id") not in ids):
                raise ValueError("Prediction row schema or ID changed")
            predictions[row["task_id"]].append(row)
    if (set(predictions) != set(ids)
            or any(len(predictions[task_id]) != receipt["row_count_by_task"][task_id]
                   for task_id in ids)
            or sum(map(len, predictions.values())) != receipt.get("total_rows")):
        raise ValueError("Prediction row count mismatch")
    return protocol, receipt, predictions, lock_sha


def fetch_results_after_start(protocol: dict) -> tuple[Path, str]:
    if not (BASE / "score_started.json").exists():
        raise AssertionError("Result access preceded score-started receipt")
    meta = protocol["result_metadata"]
    path = BASE / "results.jsonl"
    partial = BASE / "results.jsonl.part"
    if path.exists() or partial.exists():
        raise FileExistsError("Python results object already present before score")
    url = (f"https://huggingface.co/datasets/{protocol['repo']}/resolve/"
           f"{protocol['source_revision']}/{meta['path']}")
    for attempt in range(3):
        h = sha256()
        length = 0
        try:
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with partial.open("xb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            h.update(chunk)
                            length += len(chunk)
                            output.write(chunk)
            break
        except requests.RequestException:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    if length != meta["size"] or h.hexdigest() != meta["lfs_sha256"]:
        raise ValueError("Pinned Python results object size/SHA mismatch")
    partial.rename(path)
    return path, h.hexdigest()


def joined_results(path: Path, expected: set[str]) -> tuple[dict[str, dict], set[str], int]:
    rows: dict[str, dict] = {}
    duplicates: set[str] = set()
    line_count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line_count += 1
            item = json.loads(line)
            task_id = result_task_id(item)
            if task_id in expected:
                if task_id in rows:
                    duplicates.add(task_id)
                else:
                    rows[task_id] = item
    return rows, duplicates, line_count


def python_verifier(record: dict | None) -> tuple[bool | None, str]:
    if record is None:
        return None, "missing_or_duplicate_result"
    exit_code = record.get("exit_code")
    if type(exit_code) is not int:
        return None, "missing_or_invalid_exit_code"
    if exit_code != 0:
        return False, "nonzero_exit_code"
    issues = record.get("issues_count")
    pyright = record.get("pyright")
    summary = pyright.get("summary") if isinstance(pyright, dict) else None
    if (not nonnegative_int(issues) or not isinstance(pyright, dict)
            or not isinstance(pyright.get("generalDiagnostics"), list)
            or not isinstance(pyright.get("version"), str) or not pyright["version"]
            or not isinstance(summary, dict)
            or type(summary.get("filesAnalyzed")) is not int or summary["filesAnalyzed"] <= 0
            or not nonnegative_int(summary.get("errorCount"))
            or not nonnegative_int(summary.get("warningCount"))
            or not nonnegative_int(summary.get("informationCount"))
            or type(summary.get("timeInSec")) not in (int, float)
            or not math.isfinite(summary["timeInSec"])
            or summary["timeInSec"] < 0
            or any(not isinstance(item, dict) for item in pyright["generalDiagnostics"])):
        return None, "zero_exit_missing_or_invalid_pyright_output"
    import_issues = sum(item.get("rule") == "reportMissingImports"
                        for item in pyright["generalDiagnostics"])
    if import_issues != issues or summary["errorCount"] < import_issues:
        return None, "pyright_issue_count_mismatch"
    return issues == 0, "strict_python_missing_import_proxy"


def python_source(raw: bytes, task_id: str, encoding) -> dict:
    source = parse_source(raw, PARSER_ALIAS, task_id, encoding)
    for row in source["slots"]:
        row["actor"] = ACTOR
    return source


def interval(values: np.ndarray) -> dict:
    return {"lower_one_sided_95": float(np.quantile(values, .05)),
            "lower_95": float(np.quantile(values, .025)),
            "upper_95": float(np.quantile(values, .975))}


def summarize(cases: list[dict], scheduled: int, criteria: dict) -> dict:
    known = [case for case in cases if case["cost_known"]]
    point = [case for case in cases if case["eligible_input_tokens"] > 0]
    reference = np.asarray([case["reference_tokens"] for case in known], dtype=float)
    denom = float(reference.sum())
    arms = {}
    n_boot = criteria["bootstrap_draws"]
    seed = criteria["bootstrap_seed"]
    rng = np.random.default_rng(seed)
    cost_draws = rng.integers(0, len(known), size=(n_boot, len(known))) if known else None
    ref_draws = reference[cost_draws].sum(axis=1) if known else None
    valid_cost = ref_draws > 0 if known else None
    saving_draws: dict[str, np.ndarray] = {}
    for arm in ARMS:
        saved = np.asarray([case["arms"][arm]["saved_tokens"] for case in known], dtype=float)
        saving = float(saved.sum() / denom) if denom > 0 else None
        if known and valid_cost.any():
            draws = saved[cost_draws].sum(axis=1)[valid_cost] / ref_draws[valid_cost]
            saving_draws[arm] = draws
            ci = interval(draws)
        else:
            ci = None
        arms[arm] = {
            "alerts": sum(case["arms"][arm]["alert"] for case in cases),
            "saved_tokens": int(saved.sum()) if known else None,
            "pooled_saving": saving,
            "bootstrap": ci,
            "false_abort_closed_undercap": sum(
                case["arms"][arm]["false_abort_closed_undercap"] for case in cases),
            "original_success_interrupted": sum(
                case["arms"][arm]["original_verifier_success_interrupted"] for case in cases),
            "retained_observable_verifier_success": sum(
                case["arms"][arm]["retained_observable_verifier_success"] for case in cases),
        }
    cost_differences = {}
    for control in ("previous_input", "learned_minus64", "residual_minus64"):
        if control in saving_draws and "certificate" in saving_draws:
            diff = saving_draws["certificate"] - saving_draws[control]
            cost_differences[f"certificate_minus_{control}"] = {
                "point": arms["certificate"]["pooled_saving"] - arms[control]["pooled_saving"],
                "bootstrap": interval(diff),
            }
        else:
            cost_differences[f"certificate_minus_{control}"] = {"point": None, "bootstrap": None}
    actual = np.asarray([case["eligible_input_tokens"] for case in point], dtype=float)
    point_den = float(actual.sum())
    point_draws = rng.integers(0, len(point), size=(n_boot, len(point))) if point else None
    actual_draws = actual[point_draws].sum(axis=1) if point else None
    valid_point = actual_draws > 0 if point else None
    point_errors = {
        "transport": "transport_absolute_input_error",
        "previous_input": "previous_absolute_input_error",
        "learned": "learned_absolute_input_error",
        "residual": "residual_absolute_input_error",
    }
    wape = {}
    wape_draws = {}
    for name, key in point_errors.items():
        errors = np.asarray([case[key] for case in point], dtype=float)
        wape[name] = float(errors.sum() / point_den) if point_den > 0 else None
        if point and valid_point.any():
            wape_draws[name] = errors[point_draws].sum(axis=1)[valid_point] / actual_draws[valid_point]
    wape_differences = {}
    for control in ("previous_input", "learned", "residual"):
        if control in wape_draws and "transport" in wape_draws:
            delta = wape_draws[control] - wape_draws["transport"]
            wape_differences[f"{control}_minus_transport"] = {
                "point": wape[control] - wape["transport"], "bootstrap": interval(delta)}
        else:
            wape_differences[f"{control}_minus_transport"] = {"point": None, "bootstrap": None}
    closed_under = sum(case["captured_completion"] for case in cases)
    false_abort = arms["certificate"]["false_abort_closed_undercap"]
    verifier_known = sum(case["strict_verifier_success"] is not None for case in cases)
    originally_passing_under = sum(case["captured_completion"] and
                                   case["strict_verifier_success"] is True for case in cases)
    passed_under_lost = originally_passing_under - arms["certificate"]["retained_observable_verifier_success"]
    summary = {
        "scheduled_repositories": scheduled,
        "source_valid_repositories": len(cases),
        "cost_known_repositories": len(known),
        "closed_under_cap_repositories": closed_under,
        "verifier_known_repositories": verifier_known,
        "originally_passing_under_cap_repositories": originally_passing_under,
        "certificate_passing_under_cap_lost": passed_under_lost,
        "reference_tokens_known": int(denom),
        "point_eligible_repositories": len(point),
        "point_eligible_calls": sum(case["eligible_points"] for case in cases),
        "point_eligible_input_tokens": int(point_den),
        "point_wape": wape,
        "point_wape_differences": wape_differences,
        "arms": arms,
        "cost_differences": cost_differences,
        "certificate_input_lower_violations": sum(case["engineering_lower_bound_violations"] for case in cases),
        "terminal_classes": dict(__import__("collections").Counter(case["terminal_class"] for case in cases)),
        "open_censored_Tobs_lt_cap": sum(case["open_censored_Tobs_lt_cap"] for case in cases),
        "zero_false_abort_exact_one_sided_95_upper": exact_binomial_upper(false_abort, closed_under),
        "original_success_interrupted_including_overcap": arms["certificate"]["original_success_interrupted"],
        "bootstrap": {"seed": seed, "draws": n_boot,
                      "valid_cost_draws": int(valid_cost.sum()) if known else 0,
                      "valid_point_draws": int(valid_point.sum()) if point else 0},
    }
    def lower(block: dict) -> float | None:
        x = block.get("bootstrap")
        return x.get("lower_one_sided_95") if isinstance(x, dict) else None

    coverage = (len(cases) >= criteria["min_source_valid_repos"]
                and len(known) >= criteria["min_cost_known_repos"]
                and closed_under >= criteria["min_closed_undercap_repos"]
                and verifier_known >= criteria["min_verifier_known_repos"]
                and summary["point_eligible_calls"] >= criteria["min_eligible_calls"])
    material = lower(arms["certificate"]) is not None and lower(arms["certificate"]) > criteria["min_absolute_saving_lower"]
    prior_cost = (lower(cost_differences["certificate_minus_previous_input"]) is not None
                  and lower(cost_differences["certificate_minus_previous_input"]) > 0)
    prior_wape = (lower(wape_differences["previous_input_minus_transport"]) is not None
                  and lower(wape_differences["previous_input_minus_transport"]) > 0)
    quality = false_abort == 0 and passed_under_lost == 0
    empirical_bound = (summary["certificate_input_lower_violations"] <=
                       criteria["max_empirical_lower_bound_violations"])
    learned_cost = (lower(cost_differences["certificate_minus_learned_minus64"]) is not None
                    and lower(cost_differences["certificate_minus_learned_minus64"]) > 0)
    residual_cost = (lower(cost_differences["certificate_minus_residual_minus64"]) is not None
                     and lower(cost_differences["certificate_minus_residual_minus64"]) > 0)
    summary["pre_registered_checks"] = {
        "coverage": coverage, "material_absolute_saving": material,
        "cost_gain_vs_previous": prior_cost, "point_wape_gain_vs_previous": prior_wape,
        "zero_observed_closed_undercap_false_abort_and_verifier_loss": quality,
        "zero_observed_empirical_input_bound_violation": empirical_bound,
        "cost_gain_vs_learned_minus64": learned_cost,
        "cost_gain_vs_residual_minus64": residual_cost,
        "mechanism_confirmation": all((coverage, material, prior_cost, prior_wape,
                                       quality, empirical_bound)),
        "strong_control_confirmation": all((coverage, material, prior_cost, prior_wape,
                                            quality, empirical_bound, learned_cost,
                                            residual_cost)),
    }
    return summary


def score_once() -> dict:
    protocol, receipt, predictions, lock_sha = preflight()
    exclusive_json(BASE / "score_started.json", {
        "status": "started_before_nonprobe_result_fetch_or_parse",
        "protocol_sha256": lock_sha,
        "prediction_receipt_sha256": digest(BASE / "prediction_receipt.json"),
        "scorer_sha256": digest(Path(__file__)),
    })
    current_task: str | None = None
    try:
        result_path, result_sha = fetch_results_after_start(protocol)
        joined, duplicates, result_lines = joined_results(result_path, set(protocol["test_ids"]))
        encoding = tiktoken.get_encoding("o200k_base")
        cases: list[dict] = []
        source_failures: list[dict] = []
        quality_status: dict[str, str] = {}
        for task_id in protocol["test_ids"]:
            current_task = task_id
            source_path = BASE / "raw_test" / f"{task_id}.jsonl"
            raw = source_path.read_bytes()
            meta = protocol["trajectory_metadata"][task_id]
            source_sha = sha256(raw).hexdigest()
            if (len(raw) != meta["size"] or git_blob_oid(raw) != meta["blob_id"]
                    or source_sha != receipt["source_sha256"][task_id]):
                raise ValueError(f"Pinned trajectory bytes differ: {task_id}")
            try:
                source = python_source(raw, task_id, encoding)
            except Exception as exc:
                source_failures.append({"task_id": task_id, "type": type(exc).__name__,
                                        "message": str(exc)[:1000]})
                continue
            verify_predictions(predictions[task_id], source, ACTOR, task_id,
                               source_sha, 47078, 9, -64)
            result = joined.get(task_id) if task_id not in duplicates else None
            verifier, verifier_reason = python_verifier(result)
            case = score_case(source, predictions[task_id], 47078, verifier)
            case.update({"task_id": task_id, "source_sha256": source_sha,
                         "verifier_status": verifier_reason})
            quality_status[task_id] = verifier_reason
            cases.append(case)
        summary = summarize(cases, 327, protocol["success_criteria"])
        report = {
            "status": "one_shot_scored_with_predeclared_source_coverage",
            "protocol_sha256": lock_sha,
            "prediction_receipt_sha256": digest(BASE / "prediction_receipt.json"),
            "score_started_sha256": digest(BASE / "score_started.json"),
            "result_sha256": result_sha,
            "result_lines": result_lines,
            "result_join_present": len(set(joined) & set(protocol["test_ids"])),
            "result_join_duplicate_ids": sorted(duplicates),
            "result_join_missing_ids": sorted(set(protocol["test_ids"]) - set(joined)),
            "quality_status_counts": dict(__import__("collections").Counter(quality_status.values())),
            "source_failures": source_failures,
            "metrics": summary,
            "cases": cases,
        }
        exclusive_json(BASE / "score.json", report)
        return {key: value for key, value in report.items() if key != "cases"}
    except Exception as exc:
        failure = {"status": "terminal_score_failure_no_retry",
                   "protocol_sha256": lock_sha, "current_task": current_task,
                   "score_started_sha256": digest(BASE / "score_started.json"),
                   "error_type": type(exc).__name__, "error": str(exc)[:1000]}
        exclusive_json(BASE / "score_failure.json", failure)
        return failure


def fixture() -> dict:
    valid_pyright = {"version": "1.0", "generalDiagnostics": [],
                     "summary": {"filesAnalyzed": 1, "errorCount": 0,
                                 "warningCount": 0, "informationCount": 0,
                                 "timeInSec": 0.1}}
    assert python_verifier({"exit_code": 0, "issues_count": 0,
                            "pyright": valid_pyright})[0] is True
    assert python_verifier({"exit_code": 0, "issues_count": 0})[0] is None
    assert python_verifier({"exit_code": 0, "issues_count": 0,
                            "pyright": {"generalDiagnostics": [], "summary": {}}})[0] is None
    assert python_verifier({"exit_code": 0, "issues_count": 1,
                            "pyright": valid_pyright})[0] is None
    assert python_verifier({"exit_code": -127, "issues_count": 0})[0] is False
    assert abs(exact_binomial_upper(0, 64) - .045729702330762456) < 1e-12
    assert result_task_id({"repo_name": "owner/repo", "commit_sha": "a" * 40}) == "owner__repo@" + "a" * 40
    return {"status": "fixture_checks_passed", "test_or_result_access": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fixture", "score"))
    args = parser.parse_args()
    result = fixture() if args.command == "fixture" else score_once()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
