from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/envbench_four_actor_20260923"
OUT = BASE / "protocol.lock.json"
INPUTS = {
    "identity_lock": (ROOT / "workspace/experiments/envbench_budget_feasibility_20260923/jvm_bash_agent_4o_cohort.lock.json", "117b57718d307731a1049f64ec3dfd4cd05766766468bd177584ec558be3eff7"),
    "main_tree": (ROOT / "workspace/experiments/envbench_budget_independent_review_20260923/source_tree_metadata.json", "4ab03fbfc8e1ba3caaa4534f53b159482864497f1fac80c1198db124bc717a81"),
    "cross_actor_tree": (ROOT / "workspace/experiments/envbench_budget_independent_review_20260923/cross_actor_trajectory_tree_metadata.json", "4823c2e0b8384cd819efad1cb358f6daef7dea3aa06403e6e49a751f075e29ef"),
    "cross_actor_results_metadata": (ROOT / "workspace/experiments/envbench_budget_independent_review_20260923/cross_actor_result_metadata.json", "0ecad1631302aaf643f547dffed023a3b4b529ff065d6e42d59f5532080fadcb"),
    "evaluation_code_pin": (ROOT / "workspace/experiments/envbench_budget_feasibility_20260923/evaluation_code_pin.json", "0761c3b59c39bc38512780d381dbee6f7021b4f464f9144bd57be64aedd4f3ba"),
    "development_caps": (ROOT / "workspace/experiments/envbench_budget_feasibility_20260923/development_cap_candidates.json", "e204176e9be61f941cd3c0307d80e4009f3fd2068b81313d3f07967c2f17feff"),
    "cross_actor_qualification": (ROOT / "workspace/experiments/envbench_cross_actor_development_20260923/all_development_qualification.json", "226f7dbc73bb68eaf55092521ac6e874686fe97192ddf4464993c165613bf73e"),
    "cross_actor_budget_development": (ROOT / "workspace/experiments/envbench_cross_actor_development_20260923/four_actor_budget_certificate_development.json", "26b46f36eac596efd5412ab446188bd1a633be936869927290497283fdf3da1f"),
    "four_actor_paired_development": (ROOT / "workspace/experiments/envbench_cross_actor_development_20260923/four_actor_paired_development_summary.json", "54a04682bd28959f74918239c5e83ca3adb26851f286953849213e2beb2ab6ff"),
    "cross_actor_learned_control": (ROOT / "workspace/experiments/envbench_cross_actor_development_20260923/learned_cross_actor_delta_control_tool_token_v1.json", "d9caa600fb1d356b8af6e4964549eae05b3e968df4736031ff8329c8d5d30830"),
    "residual_on_transport_control": (ROOT / "workspace/experiments/envbench_cross_actor_development_20260923/residual_on_transport_v1/report.json", "f07e672a1ace2b51701a28deb419c539ef001dde33934f344c09da695bdd5032"),
    "residual_on_transport_code": (ROOT / "scripts/experiments/develop_envbench_residual_on_transport.py", "a76041e7fd388ac69798f1e5ff36aa0e3e4fbcf858e8cfe62187da41222a75a4"),
    "learned_control_reproduction": (ROOT / "workspace/experiments/envbench_cross_actor_development_20260923/learned_cross_actor_delta_control_tooltoken22_reproduction.json", "27f5d8652987a387ea761a8fb5ad63103c57e2ef5a622a8bd229ea7288a38bf4"),
    "learned_control_reproduction_code": (ROOT / "workspace/experiments/envbench_budget_independent_review_20260923/learned_cross_actor_delta_tooltoken22_reproduce.py", "28ead72a6105fcd09e7803eb630a3e3c2b10ec4782bd473aef5283397f3c6e31"),
    "main_learned_control": (ROOT / "workspace/experiments/envbench_transport_fair_budget_dev_20260923/report.json", "92a948291d54c1232bbb50c2fb0cdc8fb72804a3a6739669f927c48f206689b6"),
}
ACTORS = (
    ("jvm/bash_agent-4o", "model_stop"),
    ("jvm/bash_agent-4o-mini", "model_stop"),
    ("jvm/installamatic_agent-4o-mini", "build_end"),
    ("jvm/installamatic_agent-4o", "build_end"),
)
PREDICTOR = ROOT / "scripts/experiments/predict_envbench_four_actor_once.py"
SCORER = ROOT / "scripts/experiments/score_envbench_four_actor_once.py"
FETCHER = ROOT / "scripts/experiments/fetch_envbench_four_actor_test_sources.py"
RESULTS_FETCHER = ROOT / "scripts/experiments/fetch_envbench_four_actor_test_results.py"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_verified(name: str) -> dict:
    path, expected = INPUTS[name]
    if digest(path) != expected:
        raise ValueError(f"Pinned metadata drift: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def model_artifact(item: dict) -> dict:
    path = item["path"].replace("\\", "/")
    expected = item["sha256"]
    if digest(ROOT / path) != expected:
        raise ValueError(f"Model artifact drift: {path}")
    return {"path": path, "sha256": expected, "seed": item["seed"]}


def main() -> None:
    if OUT.exists():
        raise FileExistsError("Four-actor protocol already frozen")
    if not all(path.is_file() for path in (PREDICTOR, SCORER, FETCHER, RESULTS_FETCHER)):
        raise FileNotFoundError("Prediction, scoring, and fetch code must exist before lock")
    lock = read_verified("identity_lock")
    read_verified("main_tree")
    cross_tree = read_verified("cross_actor_tree")
    result_meta = read_verified("cross_actor_results_metadata")
    eval_pin = read_verified("evaluation_code_pin")
    caps = read_verified("development_caps")
    qualification = read_verified("cross_actor_qualification")
    budget_dev = read_verified("cross_actor_budget_development")
    paired_dev = read_verified("four_actor_paired_development")
    learned = read_verified("cross_actor_learned_control")
    residual = read_verified("residual_on_transport_control")
    if digest(INPUTS["residual_on_transport_code"][0]) != INPUTS["residual_on_transport_code"][1]:
        raise ValueError("Residual-control training code drift")
    read_verified("learned_control_reproduction")
    if digest(INPUTS["learned_control_reproduction_code"][0]) != INPUTS["learned_control_reproduction_code"][1]:
        raise ValueError("Learned-control reproduction code drift")
    read_verified("main_learned_control")
    ids = lock["untouched_test_ids"]
    if len(ids) != 132 or len(set(ids)) != 132 or set(ids) & (
        set(lock["fit_ids"]) | set(lock["calibration_ids"]) | {lock["schema_probe_excluded_id"]}
    ):
        raise ValueError("Sealed test identity drift")
    if any(item.get("test_trajectory_or_result_opened") for item in (qualification, budget_dev, learned, residual)):
        raise ValueError("Development artifact reports test access")
    if (residual.get("script_sha256") != INPUTS["residual_on_transport_code"][1]
            or residual.get("residual_feature_schema") != "tool_tokens_transport_base_residual_v1"
            or residual.get("features", [])[-2:] != ["settled_tool_content_tokens", "transport_base"]):
        raise ValueError("Residual-control feature contract drift")
    if paired_dev["repo_clusters"] != 129 or cross_tree["revision"] != lock["revision"]:
        raise ValueError("Development or source mismatch")
    if eval_pin["commit"] != "1ba2695018080df3a3606b100fc1dbfad96ea115":
        raise ValueError("Evaluation source pin mismatch")

    actor_config = {}
    for run, terminal_rule in ACTORS:
        slug = run.replace("/", "_")
        if run == "jvm/bash_agent-4o":
            metadata = {task_id: lock["trajectory_metadata"][task_id] for task_id in ids}
            cap, offset = caps["caps"]["0.5"], 9
            result = lock["result_metadata"]
        else:
            tree = cross_tree["runs"][run]["trajectory_metadata"]
            metadata = {task_id: tree[task_id] for task_id in ids}
            cap = qualification["runs"][run]["B50_candidate_fit_only_nearest_rank"]
            offset = qualification["runs"][run]["fit_only_causal_gate_framing_constant"]
            result = result_meta["runs"][run]
        model_items = [model_artifact(item) for item in learned["runs"][run]["models"]]
        residual_items = [model_artifact(item) for item in residual["runs"][run]["models"]]
        feature_schema = "tool_tokens_delta_previous"
        if len(model_items) != 2 or [item["seed"] for item in model_items] != [20260923, 20270923]:
            raise ValueError(f"Model seed contract drift: {run}")
        if len(residual_items) != 2 or [item["seed"] for item in residual_items] != [20260923, 20270923]:
            raise ValueError(f"Residual seed contract drift: {run}")
        if budget_dev["actors"][run]["B50"] != cap:
            raise ValueError(f"Budget development cap drift: {run}")
        cap_25 = budget_dev["actors"][run]["fit_only_caps"]["B25"]
        cap_75 = budget_dev["actors"][run]["fit_only_caps"]["B75"]
        if not (0 < cap_25 < cap < cap_75):
            raise ValueError(f"Budget sensitivity cap drift: {run}")
        if any(meta["path"] != f"{run}/trajectories/{task_id}.jsonl"
               or type(meta["bytes"]) is not int or meta["bytes"] <= 0
               or len(meta["git_blob_oid"]) != 40
               for task_id, meta in metadata.items()):
            raise ValueError(f"Trajectory metadata contract drift: {run}")
        actor_config[slug] = {
            "run": run,
            "cap_B50": cap,
            "cap_B25": cap_25,
            "cap_B75": cap_75,
            "fit_offset": offset,
            "lower_offset": -64,
            "gate": "one_tool_no_reset",
            "terminal_rule": terminal_rule,
            "learned_feature_schema": feature_schema,
            "learned_model_artifacts": model_items,
            "residual_feature_schema": "tool_tokens_transport_base_residual_v1",
            "residual_model_artifacts": residual_items,
            "trajectory_metadata": metadata,
            "results": {key: result[key] for key in ("path", "bytes", "lfs_sha256")},
        }
    protocol = {
        "status": "pretest_machine_protocol_lock",
        "selection_history": "Transport/gate/-64 margin and learned controls selected on 400 fit plus 132 calibration repository identities. After a reviewer-style development critique and before sealed test access, a same-arithmetic residual-on-transport control was fitted on the same fit identities, compared on calibration, and selected as the primary structural point forecast. No sealed test trajectories or result rows read",
        "repo": lock["repo"],
        "source_revision": lock["revision"],
        "one_shot_order": "freeze protocol and code; independent pretest review; retrieve and byte-hash 528 pinned test trajectories without parsing events; stream all causal predictions and seal prediction receipt; retrieve and SHA-verify four raw result blobs without JSONL parsing; create exclusive score-start receipt; then parse outcomes and score once",
        "identity_lock_sha256": INPUTS["identity_lock"][1],
        "source_tree_metadata_sha256": INPUTS["main_tree"][1],
        "cross_actor_tree_metadata_sha256": INPUTS["cross_actor_tree"][1],
        "result_metadata_sha256": INPUTS["cross_actor_results_metadata"][1],
        "evaluation_code_pin_sha256": INPUTS["evaluation_code_pin"][1],
        "development_artifacts_sha256": {name: expected for name, (_, expected) in INPUTS.items()
                                         if name not in ("identity_lock", "main_tree", "cross_actor_tree",
                                                         "cross_actor_results_metadata", "evaluation_code_pin")},
        "test_ids": ids,
        "actors": actor_config,
        "causal_gate": "prior real AI finish_reason=tool_calls; exactly one received tool; zero intervening system/human; first call and non-request sentinel abstain",
        "tokenizer": {"encoding": "o200k_base", "package": "tiktoken", "version": "0.14.0"},
        "tiktoken_version": "0.14.0",
        "forecast": "I_previous + O_previous + tokenized preceding tool content + actor fit_offset",
        "residual_forecast": "floor(max(0,I_previous+O_previous+tokenized preceding tool content+mean(two raw-scale LightGBM residual predictions)))",
        "empirical_lower_bound": "max(0, I_previous + O_previous + tool_content_tokens - 64)",
        "learned_forecast": "floor(max(0,I_previous+mean(two raw-scale LightGBM delta predictions)))",
        "arms": ["fixed_cap", "transport_minus64_same_gate", "previous_input_same_gate",
                 "learned_raw_same_gate", "learned_minus64_same_gate",
                 "residual_raw_same_gate", "residual_minus64_same_gate",
                 "current_input_same_gate_oracle", "current_input_all_slots_oracle"],
        "cost_estimand": "complete T: min(T,B); open/censored observed T>=B: B; open observed T<B: reference interval [T_obs,B], excluded from identified pooled cost",
        "quality_estimand": "closed captured T<=B false abort; nonzero integer exit_code is failure; zero exit_code with issues_count==0 and build_tool in {maven,gradle} is strict verifier success; missing or invalid necessary fields are unknown; report original successful runs stopped even if original T>B",
        "primary_aggregation": "per actor pooled saved/reference, then equal mean of four actors over common cost-identified repo identities; 10000 repo-cluster bootstrap seed 20260923 retaining four actors together",
        "primary_forecast_comparison": "four-actor equal mean input WAPE at source-gated Call Starts with settled spend strictly below actor B50: two-seed residual-on-transport point vs same-information two-seed raw input-delta model; paired repo-cluster bootstrap on raw-minus-residual difference; fixed fit-offset transport is a prespecified ablation",
        "primary_budget_comparison": "four-actor equal mean pooled saving: transport_minus64_same_gate vs previous_input_same_gate; paired repo-cluster bootstrap on candidate-minus-control difference",
        "confirmation_rules": [
            "At least 100 of 132 repositories have identifiable B50 reference cost in all four actors; all 528 identities retain a prediction or explicit source failure record",
            "Residual-on-transport point WAPE is lower than the matched raw input-delta control with a positive one-sided 95% paired repo-cluster lower bound on raw-minus-residual WAPE; fixed transport is separately reported",
            "Transport-minus64 budget saving exceeds same-gate previous-input saving with a positive one-sided 95% paired repo-cluster lower bound; its absolute saving lower bound exceeds one percentage point",
            "No observed transport-minus64 abort among closed original trajectories with T<=actor B50; report x/n and exact one-sided 95% upper bound, without claiming a one-percent population risk guarantee",
        ],
        "success_criteria": {
            "min_common_cost_identified_repos": 100,
            "min_saving_lower_one_sided_95": 0.01,
            "min_certificate_minus_previous_lower_one_sided_95": 0.0,
            "min_learned_minus_residual_wape_lower_one_sided_95": 0.0,
            "max_closed_undercap_false_abort": 0,
        },
        "secondary_comparisons": "fixed fit-offset transport point; residual raw and residual-minus64, learned raw and learned-minus64 budget guards; exact current-input same-gate and all-slots oracles; actor-specific B25/B75 certificate-versus-previous-input sensitivity from source-gated prefixes; original successful-run loss including T>B; local 528-trajectory predictor replay wall/CPU seconds from prediction receipt, including local IO and inference but excluding download and model loading; source-gate abstentions",
        "quality_statistics": "observed x/n and one-sided exact binomial 95% upper bound; zero-event bootstrap is not a 1% risk guarantee",
        "code_sha256": {"predictor": digest(PREDICTOR), "scorer": digest(SCORER),
                        "fetcher": digest(FETCHER),
                        "results_fetcher": digest(RESULTS_FETCHER),
                        "freezer": digest(Path(__file__))},
    }
    BASE.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"protocol_sha256": digest(OUT), "actors": list(actor_config),
                      "test_repositories": len(ids), "test_trajectory_contents_opened": False,
                      "run_level_results_opened": False}, indent=2))


if __name__ == "__main__":
    main()
