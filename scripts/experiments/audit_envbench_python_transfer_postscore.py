from __future__ import annotations

from collections import Counter
from hashlib import sha1, sha256
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/envbench_python_transfer_20260923"


def file_sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def equal(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def python_quality(record: dict | None) -> tuple[bool | None, str]:

    if record is None:
        return None, "missing_or_duplicate_result"
    exit_code = record.get("exit_code")
    if type(exit_code) is not int:
        return None, "missing_or_invalid_exit_code"
    if exit_code != 0:
        return False, "nonzero_exit_code"
    issues, pyright = record.get("issues_count"), record.get("pyright")
    summary = pyright.get("summary") if isinstance(pyright, dict) else None
    nonnegative = lambda value: type(value) is int and value >= 0
    if (not nonnegative(issues) or not isinstance(pyright, dict)
            or not isinstance(pyright.get("generalDiagnostics"), list)
            or not isinstance(pyright.get("version"), str) or not pyright["version"]
            or not isinstance(summary, dict)
            or type(summary.get("filesAnalyzed")) is not int or summary["filesAnalyzed"] <= 0
            or not nonnegative(summary.get("errorCount"))
            or not nonnegative(summary.get("warningCount"))
            or not nonnegative(summary.get("informationCount"))
            or type(summary.get("timeInSec")) not in (int, float)
            or not math.isfinite(summary["timeInSec"]) or summary["timeInSec"] < 0
            or any(not isinstance(item, dict) for item in pyright["generalDiagnostics"])):
        return None, "zero_exit_missing_or_invalid_pyright_output"
    import_issues = sum(item.get("rule") == "reportMissingImports"
                        for item in pyright["generalDiagnostics"])
    if import_issues != issues or summary["errorCount"] < import_issues:
        return None, "pyright_issue_count_mismatch"
    return issues == 0, "strict_python_missing_import_proxy"


def main() -> None:
    score_path = BASE / "score.json"
    if not score_path.exists():
        raise RuntimeError("One-shot score is not sealed")
    score = json.loads(score_path.read_text(encoding="utf-8"))
    protocol = json.loads((BASE / "protocol.lock.json").read_text(encoding="utf-8"))
    receipt = json.loads((BASE / "prediction_receipt.json").read_text(encoding="utf-8"))
    review = json.loads((BASE / "pretest_review_receipt.json").read_text(encoding="utf-8"))
    metrics = score["metrics"]
    ids = set(protocol["test_ids"])
    cases = score["cases"]
    failures = score["source_failures"]
    case_ids = {case["task_id"] for case in cases}
    failure_ids = {row["task_id"] for row in failures}
    assert score["status"] == "one_shot_scored_with_predeclared_source_coverage_after_network_recovery"
    assert score["protocol_sha256"] == file_sha(BASE / "protocol.lock.json")
    assert receipt["protocol_sha256"] == score["protocol_sha256"]
    assert review["protocol_sha256"] == score["protocol_sha256"]
    assert score["prediction_receipt_sha256"] == file_sha(BASE / "prediction_receipt.json")
    assert score["score_started_sha256"] == file_sha(BASE / "score_started.json")
    recovery_lock = json.loads((BASE / "recovery_protocol.lock.json").read_text(encoding="utf-8"))
    recovery_review = json.loads((BASE / "recovery_pretest_review_receipt.json").read_text(encoding="utf-8"))
    recovery_start = json.loads((BASE / "recovery_score_started.json").read_text(encoding="utf-8"))
    assert recovery_review["status"] == "pass"
    assert recovery_review["nonprobe_result_rows_parsed_by_reviewer"] is False
    assert score["recovery_lock_sha256"] == file_sha(BASE / "recovery_protocol.lock.json")
    assert score["recovery_review_sha256"] == file_sha(BASE / "recovery_pretest_review_receipt.json")
    assert score["recovery_score_started_sha256"] == file_sha(BASE / "recovery_score_started.json")
    assert recovery_review["recovery_lock_sha256"] == score["recovery_lock_sha256"]
    assert recovery_start["recovery_lock_sha256"] == score["recovery_lock_sha256"]
    assert recovery_start["result_sha256"] == score["result_sha256"]
    assert recovery_lock["recovery_script_sha256"] == file_sha(
        ROOT / "scripts/experiments/recover_envbench_python_transfer_score_once.py")
    result_path = (BASE / score["result_path"]).resolve()
    assert result_path == (BASE / recovery_lock["result_path"]).resolve()
    assert result_path.is_relative_to(BASE.resolve())
    assert result_path.stat().st_size == recovery_lock["result_size"] == protocol["result_metadata"]["size"]
    assert score["result_sha256"] == recovery_lock["result_sha256"] == protocol["result_metadata"]["lfs_sha256"]
    assert score["result_sha256"] == file_sha(result_path)
    result_counts: Counter[str] = Counter()
    result_quality: dict[str, tuple[bool | None, str]] = {}
    result_lines = 0
    with result_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            result_lines += 1
            repo, commit = row["repo_name"], row["commit_sha"]
            assert isinstance(repo, str) and repo.count("/") == 1
            assert isinstance(commit, str) and len(commit) == 40
            owner, name = repo.split("/")
            task_id = f"{owner}__{name}@{commit}"
            result_counts[task_id] += 1
            if task_id in ids and task_id not in result_quality:
                result_quality[task_id] = python_quality(row)
    assert score["result_lines"] == result_lines
    assert score["result_join_present"] == len(result_quality)
    assert score["result_join_duplicate_ids"] == sorted(
        task_id for task_id in ids if result_counts[task_id] > 1)
    assert score["result_join_missing_ids"] == sorted(ids - set(result_quality))
    off_roster_result_ids = sorted(set(result_counts) - ids)
    assert len(cases) == len(case_ids)
    assert len(failures) == len(failure_ids)
    assert not case_ids & failure_ids
    assert case_ids | failure_ids == ids
    assert metrics["source_valid_repositories"] == len(cases)
    for case in cases:
        task_id = case["task_id"]
        raw = (BASE / "raw_test" / f"{task_id}.jsonl").read_bytes()
        source_sha = sha256(raw).hexdigest()
        meta = protocol["trajectory_metadata"][task_id]
        blob = sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        assert len(raw) == meta["size"] and blob == meta["blob_id"]
        assert source_sha == case["source_sha256"] == receipt["source_sha256"][task_id]
    assert score["quality_status_counts"] == dict(Counter(
        case["verifier_status"] for case in cases))
    for case in cases:
        task_id = case["task_id"]
        expected_quality = (None, "missing_or_duplicate_result") if result_counts[task_id] != 1 else result_quality[task_id]
        assert (case["strict_verifier_success"], case["verifier_status"]) == expected_quality
    assert metrics["terminal_classes"] == dict(Counter(
        case["terminal_class"] for case in cases))
    assert metrics["open_censored_Tobs_lt_cap"] == sum(
        case["open_censored_Tobs_lt_cap"] for case in cases)

    known = [case for case in cases if case["cost_known"]]
    reference = sum(case["reference_tokens"] for case in known)
    assert metrics["cost_known_repositories"] == len(known)
    assert metrics["reference_tokens_known"] == reference
    assert metrics["closed_under_cap_repositories"] == sum(case["captured_completion"] for case in cases)
    assert metrics["verifier_known_repositories"] == sum(case["strict_verifier_success"] is not None
                                                         for case in cases)
    assert metrics["certificate_input_lower_violations"] == sum(
        case["engineering_lower_bound_violations"] for case in cases)
    assert metrics["point_eligible_calls"] == sum(case["eligible_points"] for case in cases)
    assert metrics["point_eligible_input_tokens"] == sum(case["eligible_input_tokens"] for case in cases)
    assert metrics["point_eligible_repositories"] == sum(
        case["eligible_input_tokens"] > 0 for case in cases)
    for case in cases:
        total = case["total_observed_tokens"]
        closed = case["terminal_class"] in ("model_stop_closed", "build_closed")
        known_cost = closed or total >= 47078
        assert case["cost_known"] == known_cost
        assert case["reference_tokens"] == (min(total, 47078) if known_cost else None)
        assert case["captured_completion"] == (closed and total <= 47078)
        assert case["open_censored_Tobs_lt_cap"] == (not closed and total < 47078)
    for arm, reported in metrics["arms"].items():
        assert reported["alerts"] == sum(case["arms"][arm]["alert"] for case in cases)
        assert reported["saved_tokens"] == sum(case["arms"][arm]["saved_tokens"] for case in known)
        assert equal(reported["pooled_saving"], reported["saved_tokens"] / reference)
        assert reported["false_abort_closed_undercap"] == sum(
            case["arms"][arm]["false_abort_closed_undercap"] for case in cases)
        assert reported["original_success_interrupted"] == sum(
            case["arms"][arm]["original_verifier_success_interrupted"] for case in cases)
        assert reported["retained_observable_verifier_success"] == sum(
            case["arms"][arm]["retained_observable_verifier_success"] for case in cases)
        for case in cases:
            decision = case["arms"][arm]
            expected = (case["reference_tokens"] - decision["spent_at_stop"]
                        if decision["alert"] and case["cost_known"]
                        else 0 if case["cost_known"] else None)
            assert decision["saved_tokens"] == expected
    point = [case for case in cases if case["eligible_input_tokens"] > 0]
    point_den = sum(case["eligible_input_tokens"] for case in point)
    error_keys = {"transport": "transport_absolute_input_error",
                  "previous_input": "previous_absolute_input_error",
                  "learned": "learned_absolute_input_error",
                  "residual": "residual_absolute_input_error"}
    for name, key in error_keys.items():
        assert equal(metrics["point_wape"][name],
                     sum(case[key] for case in point) / point_den)
    draw_count = protocol["success_criteria"]["bootstrap_draws"]
    seed = protocol["success_criteria"]["bootstrap_seed"]
    rng = np.random.default_rng(seed)
    cost_draws = rng.integers(0, len(known), size=(draw_count, len(known)))
    ref_array = np.asarray([case["reference_tokens"] for case in known])
    ref_draws = ref_array[cost_draws].sum(axis=1)
    valid_cost = ref_draws > 0
    assert metrics["bootstrap"]["valid_cost_draws"] == int(valid_cost.sum())
    cost_draw_rates = {}
    for arm, reported in metrics["arms"].items():
        saved = np.asarray([case["arms"][arm]["saved_tokens"] for case in known])
        rate = saved[cost_draws].sum(axis=1)[valid_cost] / ref_draws[valid_cost]
        cost_draw_rates[arm] = rate
        ci = reported["bootstrap"]
        assert equal(ci["lower_one_sided_95"], float(np.quantile(rate, .05)))
        assert equal(ci["lower_95"], float(np.quantile(rate, .025)))
        assert equal(ci["upper_95"], float(np.quantile(rate, .975)))
    for control in ("previous_input", "learned_minus64", "residual_minus64"):
        name = f"certificate_minus_{control}"
        delta = cost_draw_rates["certificate"] - cost_draw_rates[control]
        difference = metrics["cost_differences"][name]
        assert equal(difference["point"],
                     metrics["arms"]["certificate"]["pooled_saving"]
                     - metrics["arms"][control]["pooled_saving"])
        ci = difference["bootstrap"]
        assert equal(ci["lower_one_sided_95"], float(np.quantile(delta, .05)))
        assert equal(ci["lower_95"], float(np.quantile(delta, .025)))
        assert equal(ci["upper_95"], float(np.quantile(delta, .975)))
    point_draws = rng.integers(0, len(point), size=(draw_count, len(point)))
    actual = np.asarray([case["eligible_input_tokens"] for case in point])
    actual_draws = actual[point_draws].sum(axis=1)
    valid_point = actual_draws > 0
    assert metrics["bootstrap"]["valid_point_draws"] == int(valid_point.sum())
    errors = {name: np.asarray([case[key] for case in point])
              for name, key in error_keys.items()}
    transport = errors["transport"][point_draws].sum(axis=1)[valid_point] / actual_draws[valid_point]
    for control in ("previous_input", "learned", "residual"):
        other = errors[control][point_draws].sum(axis=1)[valid_point] / actual_draws[valid_point]
        difference = metrics["point_wape_differences"][f"{control}_minus_transport"]
        assert equal(difference["point"],
                     metrics["point_wape"][control] - metrics["point_wape"]["transport"])
        delta = other - transport
        ci = difference["bootstrap"]
        assert equal(ci["lower_one_sided_95"], float(np.quantile(delta, .05)))
        assert equal(ci["lower_95"], float(np.quantile(delta, .025)))
        assert equal(ci["upper_95"], float(np.quantile(delta, .975)))
    checks = metrics["pre_registered_checks"]
    criteria = protocol["success_criteria"]
    lower = lambda block: block["bootstrap"]["lower_one_sided_95"]
    certificate = metrics["arms"]["certificate"]
    closed_under = metrics["closed_under_cap_repositories"]
    originally_passing_under = sum(
        case["captured_completion"] and case["strict_verifier_success"] is True
        for case in cases)
    passed_under_lost = originally_passing_under - certificate["retained_observable_verifier_success"]
    assert metrics["originally_passing_under_cap_repositories"] == originally_passing_under
    assert metrics["certificate_passing_under_cap_lost"] == passed_under_lost
    false_abort = certificate["false_abort_closed_undercap"]
    assert equal(metrics["zero_false_abort_exact_one_sided_95_upper"],
                 1 - .05 ** (1 / closed_under) if false_abort == 0 else
                 metrics["zero_false_abort_exact_one_sided_95_upper"])
    recomputed_checks = {
        "coverage": (len(cases) >= criteria["min_source_valid_repos"]
                     and len(known) >= criteria["min_cost_known_repos"]
                     and closed_under >= criteria["min_closed_undercap_repos"]
                     and metrics["verifier_known_repositories"] >= criteria["min_verifier_known_repos"]
                     and metrics["point_eligible_calls"] >= criteria["min_eligible_calls"]),
        "material_absolute_saving": lower(certificate) > criteria["min_absolute_saving_lower"],
        "cost_gain_vs_previous": lower(metrics["cost_differences"]["certificate_minus_previous_input"]) > 0,
        "point_wape_gain_vs_previous": lower(metrics["point_wape_differences"]["previous_input_minus_transport"]) > 0,
        "zero_observed_closed_undercap_false_abort_and_verifier_loss": false_abort == 0 and passed_under_lost == 0,
        "zero_observed_empirical_input_bound_violation": (
            metrics["certificate_input_lower_violations"] <= criteria["max_empirical_lower_bound_violations"]),
        "cost_gain_vs_learned_minus64": lower(metrics["cost_differences"]["certificate_minus_learned_minus64"]) > 0,
        "cost_gain_vs_residual_minus64": lower(metrics["cost_differences"]["certificate_minus_residual_minus64"]) > 0,
    }
    core = ("coverage", "material_absolute_saving", "cost_gain_vs_previous",
            "point_wape_gain_vs_previous", "zero_observed_closed_undercap_false_abort_and_verifier_loss",
            "zero_observed_empirical_input_bound_violation")
    recomputed_checks["mechanism_confirmation"] = all(recomputed_checks[key] for key in core)
    recomputed_checks["strong_control_confirmation"] = (
        recomputed_checks["mechanism_confirmation"]
        and recomputed_checks["cost_gain_vs_learned_minus64"]
        and recomputed_checks["cost_gain_vs_residual_minus64"])
    assert checks == recomputed_checks
    output = {"status": "independent_postscore_arithmetic_passed",
              "score_sha256": file_sha(score_path),
              "protocol_sha256": score["protocol_sha256"],
              "scheduled": len(ids), "source_valid": len(cases),
              "source_failures": len(failures),
              "cost_known": len(known),
              "result_file_bytes": result_path.stat().st_size,
              "result_lines": result_lines,
              "result_join_present": len(result_quality),
              "result_join_missing_ids": score["result_join_missing_ids"],
              "off_roster_result_ids": off_roster_result_ids,
              "quality_status_counts": score["quality_status_counts"],
              "quality_unknown_count": sum(case["strict_verifier_success"] is None for case in cases),
              "point_eligible_calls": metrics["point_eligible_calls"],
              "point_wape": metrics["point_wape"],
              "arms": {arm: {key: values[key] for key in ("pooled_saving", "bootstrap",
                                                            "false_abort_closed_undercap")}
                       for arm, values in metrics["arms"].items()},
              "cost_differences": metrics["cost_differences"],
              "zero_false_abort_exact_one_sided_95_upper": metrics["zero_false_abort_exact_one_sided_95_upper"],
              "result_sha256_verified": True,
              "result_join_and_strict_quality_recomputed": True,
              "source_blob_and_censoring_recomputed": True,
              "case_and_cluster_bootstrap_recomputed": True,
              "pre_registered_checks": checks}
    path = BASE / "postscore_arithmetic_audit.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
