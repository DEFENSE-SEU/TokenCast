from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

import tiktoken
import lightgbm as lgb
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/envbench_python_transfer_20260923"
AUDIT = ROOT / "workspace/experiments/envbench_python_second_source_20260923/source_feasibility.json"
JVM_PROTOCOL = ROOT / "workspace/experiments/envbench_four_actor_20260923/protocol.lock.json"
SOURCE = ROOT / "workspace/experiments/envbench_source_current_20260923"
TASK_RE = re.compile(r"^[^@/\\]+__[^@/\\]+@[0-9a-f]{40}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    archives = {"protocol.lock.pretest-v1.json", "protocol.lock.pretest-v2.json",
                "protocol.lock.pretest-v3.json", "protocol.lock.pretest-v4.json"}
    if BASE.exists() and {item.name for item in BASE.iterdir()} - archives:
        raise FileExistsError("Python transfer directory already contains a lock or test artifact")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    jvm = json.loads(JVM_PROTOCOL.read_text(encoding="utf-8"))
    if (audit.get("revision") != "4ffd02bed0f39fd87bffe3be7edf2440aba16cd1"
            or audit.get("run") != "python/bash_agent-4o"
            or audit.get("trajectory_count") != 328
            or audit.get("candidate_after_probe_exclusion") != 327
            or audit.get("overlap_with_jvm_exact_ids") != 0
            or audit.get("overlap_with_jvm_repository_names") != 0):
        raise ValueError("Source audit does not establish the specified independent roster")
    probe = audit["excluded_probe"]["task_id"]
    metadata = audit["trajectory_metadata"]
    ids = sorted(set(metadata) - {probe})
    if len(ids) != 327 or any(TASK_RE.fullmatch(task_id) is None for task_id in ids):
        raise ValueError("Python test IDs malformed")
    for task_id in ids:
        row = metadata[task_id]
        if (row.get("path") != f"python/bash_agent-4o/trajectories/{task_id}.jsonl"
                or type(row.get("size")) is not int or row["size"] <= 0
                or not isinstance(row.get("blob_id"), str)
                or HEX40.fullmatch(row["blob_id"]) is None):
            raise ValueError(f"Pinned source metadata malformed: {task_id}")
    result = audit["results_metadata"]
    if (result.get("path") != "python/bash_agent-4o/results.jsonl"
            or result.get("size") != 182813874
            or result.get("lfs_sha256") !=
            "96675c2e4db885f35e34d7d5748bde5be0d92f3e53f5bdb7ad194c9281710ee7"):
        raise ValueError("Python result metadata changed")
    jvm_actor = jvm["actors"]["jvm_bash_agent-4o"]
    imported_jvm_code_sha256 = {
        "predictor": digest(ROOT / "scripts/experiments/predict_envbench_four_actor_once.py"),
        "scorer": digest(ROOT / "scripts/experiments/score_envbench_four_actor_once.py"),
    }
    if imported_jvm_code_sha256 != {key: jvm.get("code_sha256", {}).get(key)
                                    for key in ("predictor", "scorer")}:
        raise ValueError("Imported JVM source no longer matches its locked protocol")
    if (jvm_actor["cap_B50"], jvm_actor["fit_offset"], jvm_actor["lower_offset"],
            jvm_actor["gate"]) != (47078, 9, -64, "one_tool_no_reset"):
        raise ValueError("JVM-locked transfer rule changed")
    for key in ("learned_model_artifacts", "residual_model_artifacts"):
        for model in jvm_actor[key]:
            if digest(ROOT / model["path"]) != model["sha256"]:
                raise ValueError(f"JVM model artifact changed: {model['path']}")
    source_commit = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if source_commit != "1ba2695018080df3a3606b100fc1dbfad96ea115":
        raise ValueError("Official EnvBench code pin changed")
    code_paths = (
        "inference/src/agents/python/agent.py",
        "inference/src/agents/python/prompts.py",
        "inference/src/utils/messages_info.py",
        "evaluation/main.py",
        "evaluation/scripts/python_build.sh",
    )
    protocol = {
        "status": "zero_python_tuning_transfer_locked",
        "repo": audit["dataset"],
        "source_revision": audit["revision"],
        "run": audit["run"],
        "actor": "python_bash_agent-4o",
        "jvm_source_actor": "jvm_bash_agent-4o",
        "jvm_protocol_sha256": digest(JVM_PROTOCOL),
        "imported_jvm_code_sha256": imported_jvm_code_sha256,
        "transitive_feature_code_sha256": digest(
            ROOT / "scripts/experiments/develop_envbench_jvm_budget_models.py"),
        "source_audit_sha256": digest(AUDIT),
        "excluded_schema_probe_id": probe,
        "test_ids": ids,
        "trajectory_metadata": {task_id: metadata[task_id] for task_id in ids},
        "result_metadata": result,
        "source_code_commit": source_commit,
        "source_code_sha256": {path: digest(SOURCE / path) for path in code_paths},
        "policy": {"cap": 47078, "fit_offset": 9, "lower_offset": -64,
                   "gate": "one_tool_no_reset"},
        "tokenizer": "o200k_base",
        "tiktoken_version": tiktoken.__version__,
        "numpy_version": np.__version__,
        "lightgbm_version": lgb.__version__,
        "learned_feature_schema": "tool_tokens_delta_previous",
        "learned_model_artifacts": jvm_actor["learned_model_artifacts"],
        "residual_feature_schema": "tool_tokens_transport_base_residual_v1",
        "residual_model_artifacts": jvm_actor["residual_model_artifacts"],
        "causal_clock": "after prior AI usage and tool messages, before current AI usage",
        "cost_estimand": {
            "unit": "nominal_provider_input_plus_output_tokens",
            "cache_read_is_subset_of_input": True,
            "reference_closed": "min(T_observed,47078)",
            "reference_open_Tobs_ge_B": 47078,
            "reference_open_Tobs_lt_B": "unknown_excluded_from_cost_rate",
            "stop_cost": "spent_before_first_eligible_hit",
        },
        "quality_estimand": {
            "terminal_closed": "last_real_ai_finish_reason_stop_and_commands_history_tail",
            "step_limit_sentinel": "right_censored_not_provider_call",
            "strict_python_proxy": "exit_code_0_and_pyright_generalDiagnostics_and_summary_valid_and_recomputed_reportMissingImports_eq_issues_count_0",
            "missing_or_duplicate_result": "quality_unknown",
            "closed_under_cap_false_abort": "any_stop_on_model_stop_closed_with_T_observed_le_47078",
            "original_success_interrupted": "reported_including_original_success_above_cap",
        },
        "arms": ["certificate", "previous_input", "learned_raw", "learned_minus64",
                 "residual_raw", "residual_minus64", "exact_current_input_same_gate_oracle",
                 "exact_current_input_all_slots_oracle"],
        "primary_comparisons": {
            "point": "WAPE_previous_input_minus_WAPE_transport_on_same_eligible_calls",
            "cost": "pooled_saving_certificate_minus_previous_input_on_common_known_cost_repos",
            "strong_controls": "certificate_minus_learned_minus64_and_residual_minus64",
            "oracle": "privileged_exact_current_input_separate",
        },
        "success_criteria": {
            "min_source_valid_repos": 260,
            "min_cost_known_repos": 260,
            "min_closed_undercap_repos": 100,
            "min_verifier_known_repos": 260,
            "min_eligible_calls": 500,
            "min_absolute_saving_lower": .01,
            "max_closed_undercap_false_abort": 0,
            "max_empirical_lower_bound_violations": 0,
            "bootstrap_seed": 20260923,
            "bootstrap_draws": 10000,
            "one_sided_lower_quantile": .05,
            "confidence_reporting": "repo_cluster_bootstrap_and_exact_binomial_for_false_aborts",
            "mechanism_confirmation": "all_coverage_and_material_and_certificate_gt_previous_cost_and_transport_lt_previous_WAPE_and_zero_observed_undercap_loss_and_zero_input_bound_violation",
            "strong_control_confirmation": "mechanism_confirmation_and_certificate_gt_learned_minus64_and_residual_minus64_cost",
        },
        "prediction_order": "exclusive_prediction_started_then_fetch_327_sha_verified_trajectories_then_emit_before_current_usage_then_lock_predictions",
        "score_order": "verify_lock_predictions_then_exclusive_score_started_then_fetch_sha_verified_results_then_score_once",
        "test_data_access_before_lock": "only_excluded_probe_and_metadata;_first_HTTP_range_larger_than_one_line_but_only_first_JSON_row_parsed",
        "code_sha256": {
            "predictor": digest(ROOT / "scripts/experiments/predict_envbench_python_transfer_once.py"),
            "scorer": digest(ROOT / "scripts/experiments/score_envbench_python_transfer_once.py"),
            "source_feasibility_audit": digest(ROOT / "scripts/experiments/audit_envbench_python_second_source.py"),
        },
    }
    BASE.mkdir(parents=True, exist_ok=True)
    lock_path = BASE / "protocol.lock.json"
    with lock_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(protocol, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(lock_path)
    print(digest(lock_path))


if __name__ == "__main__":
    main()
