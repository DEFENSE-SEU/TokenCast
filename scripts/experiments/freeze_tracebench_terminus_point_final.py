import importlib.metadata
import json
import platform
from pathlib import Path

from tracebench_terminus_preflight import ROOT, WORK, sha256
from tracebench_terminus_point_runtime import ALLOWED_ROUTES


CODE_FILES = {
    "final_freezer": "scripts/experiments/freeze_tracebench_terminus_point_final.py",
    "qualification": "scripts/experiments/qualify_tracebench_terminus_development.py",
    "original_fit": "scripts/experiments/fit_tracebench_terminus_point.py",
    "visible_fit": "scripts/experiments/fit_tracebench_visible_transport.py",
    "runtime": "scripts/experiments/tracebench_terminus_point_runtime.py",
    "preflight": "scripts/experiments/tracebench_terminus_preflight.py",
    "test_download": "scripts/experiments/download_tracebench_terminus_test.py",
    "test_predict": "scripts/experiments/predict_tracebench_terminus_point.py",
    "test_score": "scripts/experiments/score_tracebench_terminus_point.py",
    "development_smoke": "scripts/experiments/smoke_tracebench_terminus_runtime.py",
}
MODEL_FILES = {
    "direct_visible": "workspace/experiments/tracebench_cross_harness_feasibility_20260923/point_visible_fit/direct_visible.txt",
    "residual_visible": "workspace/experiments/tracebench_cross_harness_feasibility_20260923/point_visible_fit/residual_visible.txt",
    "affine": "workspace/experiments/tracebench_cross_harness_feasibility_20260923/point_fit/affine.json",
    "offsets": "workspace/experiments/tracebench_cross_harness_feasibility_20260923/point_visible_fit/offsets.json",
}


def main() -> None:
    output = WORK / "terminus_point_protocol.final.lock.json"
    if output.exists():
        raise RuntimeError("Final lock already exists")
    forbidden_test_paths = (
        WORK / "test_archives", WORK / "test_download_receipt.json",
        WORK / "point_prediction_started.json", WORK / "point_predictions.jsonl",
        WORK / "point_prediction_coverage.jsonl", WORK / "point_prediction_sealed.json",
        WORK / "point_score_started.json", WORK / "point_scored_labels.jsonl",
        WORK / "point_score.json",
    )
    if any(path.exists() for path in forbidden_test_paths):
        raise RuntimeError("Test content or run markers exist before final lock")
    draft_path = WORK / "terminus_point_protocol.predev.lock.json"
    index_path = WORK / "terminus_artifact_index.metadata.json"
    smoke_path = WORK / "runtime_development_smoke_v5.json"
    visible_report_path = WORK / "point_visible_fit/visible_fit_report.json"
    original_report_path = WORK / "point_fit/fit_report.json"
    no_go_path = WORK / "development_visible_no_go.md"
    draft = json.loads(draft_path.read_text())
    visible = json.loads(visible_report_path.read_text())
    original = json.loads(original_report_path.read_text())
    smoke = json.loads(smoke_path.read_text())
    if sha256(index_path) != draft["artifact_index_sha256"]:
        raise RuntimeError("Metadata index changed")
    routes = [route for allowed in ALLOWED_ROUTES.values() for route in allowed]
    if len(routes) != len(set(routes)) or set(ALLOWED_ROUTES) != set(draft["planned_test_trajectories_per_model"]):
        raise RuntimeError("Old-identity model routes are ambiguous")
    if smoke["status"] != "old_identity_runtime_gate_match" or smoke["archives"] != 356 or smoke["pre_gate_points"] != 14048:
        raise RuntimeError("Old-identity runtime smoke did not pass")
    if visible["calibration"]["n"] != 3902 or visible["calibration"]["p_minus_64_violations"]["frozen_visible_transport"] != 31:
        raise RuntimeError("Development decision changed")
    if visible["calibration"]["macro_wape"]["visible_fit_transport_residual"] >= visible["calibration"]["macro_wape"]["visible_fit_direct_delta"]:
        raise RuntimeError("Visible relation did not beat direct learned control on calibration")
    if visible["code_sha256"]["visible_fit"] != sha256(ROOT / CODE_FILES["visible_fit"]) or visible["code_sha256"]["qualification"] != sha256(ROOT / CODE_FILES["qualification"]):
        raise RuntimeError("Visible model training source changed")
    for key, path in MODEL_FILES.items():
        expected = original["model_sha256"]["affine"] if key == "affine" else visible["model_sha256"][key]
        if sha256(ROOT / path) != expected:
            raise RuntimeError(f"Frozen model differs from fit report: {key}")
    lock = {
        "status": "final_point_only_locked",
        "source": "Contextbench/Tracebench full",
        "source_revision": draft["source_revision"],
        "manifest_sha256": draft["manifest_sha256"],
        "artifact_index_sha256": draft["artifact_index_sha256"],
        "actor": "Terminus2",
        "excluded_probe_task_id": draft["excluded_probe_task_id"],
        "fit_task_ids": draft["fit_task_ids"],
        "calibration_task_ids": draft["calibration_task_ids"],
        "test_task_ids": draft["test_task_ids"],
        "planned_test_trajectories_per_model": draft["planned_test_trajectories_per_model"],
        "minimum_test_coverage": draft["minimum_test_coverage"],
        "old_identity_development": {
            "fit_task_ids": 60, "calibration_task_ids": 26, "archives": 356,
            "pre_gate_points": 14048, "calibration_scored_points": 3902,
            "visible_residual_macro_wape": visible["calibration"]["macro_wape"]["visible_fit_transport_residual"],
            "same_info_direct_macro_wape": visible["calibration"]["macro_wape"]["visible_fit_direct_delta"],
            "visible_fixed_p_minus_64_violations": 31,
            "budget_lower_bound_decision": "no_go_for_hard_bound_or_spend_claim",
        },
        "target": "Retained successful LiteLLM original_response.usage.prompt_tokens of logical request j; response-reported count only, with unobserved retries and provider billing out of scope",
        "information_clock": "Retrospective hypothetical request-start view reconstructed from post-API debug.json; previous retained response is settled; current request is projected from a strict allowlist; current response usage is parsed only after its forecast is appended, flushed and fsynced",
        "request_provenance": "Older public Terminal-Bench source constructs messages and calls litellm.completion before post_api_call logging; runtime checkout, exact HTTP body and backend model identity are unverified. complete_input_dict.model is a LiteLLM route identifier checked against old-identity allowlists.",
        "allowed_litellm_model_routes_by_manifest_model": {model: sorted(routes) for model, routes in ALLOWED_ROUTES.items()},
        "prediction_pre_gate": [
            "Pinned test artifact identity, size and LFS SHA must match the metadata index",
            "Consecutive episode j-1,j; previous retained response has one choice with finish_reason=stop, role=assistant and text content, nonnegative response-reported input/output usage and consistent total when present",
            "Current complete_input_dict has exactly model,messages,temperature,extra_body; extra_body is empty; messages are role/content only with strings or a single text block and optional ephemeral cache_control; recursively reject response/usage/token-count/cost/credential keys",
            "complete_input_dict.messages exactly equals debug.messages; unknown fields or noncanonical JSON are ineligible",
            "complete_input_dict.model belongs to the frozen old-identity route set for the manifest model and is unchanged within each adjacent request pair; actual backend identity is not inferred from this route alone",
            "For history-chain comparison only, normalize one text content block to its string and remove its cache_control; messages_j minus the appended assistant/user pair must equal normalized messages_(j-1)",
            "Assistant role/content appended to j must exactly equal the previous retained returned choice content after the same cache normalization; other nonempty dropped response fields except provider-private reasoning are ineligible",
            "Final user text after narrow cache normalization exactly equals current prompt.txt",
            "Adjacent request starts must both carry timezones or both be naive source-local ISO timestamps; previous duration is finite nonnegative and its summed time is no later than current start. Mixed, missing or overflowing clocks make the point ineligible. Prior response/call IDs are not repeated among earlier settled records",
            "No current response or result field enters this gate or any forecast feature",
        ],
        "postprediction_score_gate": [
            "After current j forecast is appended/flushed/fsynced, parse current retained response usage; a single finish_reason=stop assistant text choice, nonnegative prompt/completion counts, consistent total and cache/reasoning bucket bounds when present, and nonduplicated response/call IDs yield a score label",
            "All pre-gate forecasts remain in coverage; later missing/failed/duplicate responses are predicted-without-label, never removed retroactively",
        ],
        "request_proxy": "o200k_base text length of canonical JSON complete_input_dict; includes model, roles, full messages, options and any cache text blocks; tokenizer is not provider-exact",
        "visible_output_proxy": "o200k_base length of the assistant text actually appended in current recorded messages; GPT-5 hidden reasoning is excluded from this visible content, while O_previous remains available to the learned controls",
        "fixed_visible_formula": "I_prev+O_visible+o200k(prompt_j)+9",
        "primary_arm": "visible_fit_transport_residual: fit-only L1 LightGBM estimates I_j−(I_prev+O_visible+o200k(prompt_j)); clipped nonnegative",
        "comparator_arms": [
            "visible_fit_direct_delta", "frozen_visible_transport", "fit_offset_visible_transport",
            "original_frozen_transport", "previous_input", "full_request_affine",
        ],
        "additional_reported_arm": "no_new_observation_visible",
        "same_information": "Both learned arms use I_prev,O_prev,O_visible,current prompt text proxy, episode index, full canonical request proxy, message count and model indicators. Direct predicts I_j−I_prev; residual predicts deviation from visible transport. Each fit task has total weight one.",
        "fit_only_controls": "Two 180-round L1 LightGBM models with learning rate 0.05, 15 leaves, min leaf 40, L2 1 and seed 20260923; full-request affine is a separate per-model nonnegative slope/free intercept fit; visible offset is per-model task-weighted fit median residual",
        "development_choice": "One visible-output correction was selected on old fit/cal before any new-identity test access. The learned residual was named primary because its old calibration macro-WAPE beat same-information direct; all structural and learned controls remain mandatory comparisons.",
        "point_metric": "Five-model equal-weight mean of per-model WAPE on the exact same postprediction label-eligible calls, with pooled WAPE, task-equal MAE, per-model WAPE and pre-gate/score-gate coverage",
        "bootstrap": "After minimum coverage and positive per-model target mass, draw 10000 paired task-ID cluster resamples, seed 20260923, preserving all model trajectories for each sampled task; report two-sided 95% CIs and one-sided 95% lower bounds for comparator macro-WAPE minus primary. If any draw lacks positive mass for a model, mark CI unavailable and success false without interrupting score sealing",
        "common_model_sensitivity": "Repeat metrics and paired task bootstrap on the 78 metadata-planned task IDs with all five model slots; explicitly report scored and missing model-task slots. Sparse/zero-mass draws make the sensitivity CI unavailable without interrupting score sealing",
        "success_criterion": "All minimum coverage gates pass and the paired task-bootstrap one-sided 95% lower improvement bound is strictly positive for each of the six mandatory comparators; failure of any comparison is reported without changing arms or coverage",
        "p_minus_64_diagnostic": "Report observed violations for original frozen, fixed visible and visible residual forecasts only as finite-sample diagnostics. Old calibration already had 31/3902 fixed-visible violations; no hard-budget, spend-saving or universal lower-bound claim is made.",
        "result_policy": "No results.json, solved, task-quality or currency score is parsed by prediction or point scorer; raw traces are retained locally only",
        "independent_review_disclosure": "After development and the first lock, a separate reviewer saw public dataset-page preview solved values for a different actor on some same-named tasks while checking rights. No Terminus2 test archive/result was accessed and no displayed values entered code, fit, gates or metrics.",
        "stop_conditions": [
            "Independent reviewer has not written a pass receipt for this exact final protocol hash",
            "Any code/model/dependency/artifact hash or test task identity mismatch",
            "Any new request schema fails the frozen allowlist: mark that point ineligible; do not extend the gate after test access",
            "Minimum test coverage fails: mark predictive transfer inconclusive without replacing task IDs or tuning",
        ],
        "run_order": "Independent pass receipt; download/hash test archive bytes without extraction; create prediction-start marker; each episode request view→pre-gate→forecast fsync→current response view; seal all predictions; create score-start marker; reread responses for label availability and metrics. Test results/solved are never parsed.",
        "source_rights": "Dataset card has no explicit dataset license; use local analysis and aggregate reporting, with no raw trajectory redistribution",
        "paid_model_calls": 0,
        "test_content_read_before_lock": False,
        "predev_protocol_sha256": sha256(draft_path),
        "development_smoke_sha256": sha256(smoke_path),
        "visible_fit_report_sha256": sha256(visible_report_path),
        "original_fit_report_sha256": sha256(original_report_path),
        "budget_no_go_report_sha256": sha256(no_go_path),
        "code_files": CODE_FILES,
        "code_sha256": {key: sha256(ROOT / path) for key, path in CODE_FILES.items()},
        "model_files": MODEL_FILES,
        "model_sha256": {key: sha256(ROOT / path) for key, path in MODEL_FILES.items()},
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "lightgbm", "tiktoken", "zstandard", "requests")},
        "python_version": platform.python_version(),
        "independent_pass_receipt_expected": {"status": "pass", "protocol_sha256": "SHA256 of this final protocol file"},
    }
    with output.open("x", encoding="utf-8") as out:
        out.write(json.dumps(lock, indent=2, ensure_ascii=True) + "\n")
    print(f"final point-only protocol SHA256 {sha256(output)}")


if __name__ == "__main__":
    main()
