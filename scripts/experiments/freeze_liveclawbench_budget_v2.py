from __future__ import annotations

import json
from pathlib import Path

from scripts.experiments.calibrate_liveclawbench_budget import BASE, MANIFEST, SOURCE, digest


ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "workspace/experiments/liveclawbench_budget_v2_20260923"
OUT = OUTDIR / "lock.json"
PROTOCOL = ROOT / "docs/local/liveclawbench-call-start-budget-v2-prescore-protocol-2026-09-23.md"
RUNNER = ROOT / "scripts/experiments/run_liveclawbench_budget_v2.py"
CORE = BASE / "v2_core_ab_calibration.json"
MODEL_CAL = BASE / "v2_model_controls_ab_calibration.json"
QFLOOR = BASE / "decision_risk_qfloor_development.json"
DEV_DRY = BASE / "v2_real_model_dev_dry_run.json"
EXTERNAL = ROOT / "workspace/experiments/exgentic_next_input_baselines_20260923_v2"
DOMAIN = ROOT / "workspace/experiments/liveclawbench_next_input_control_20260923"
REMAINING = ROOT / "workspace/experiments/exgentic_v2_context_exposure_prefit_20260923_v2/models"
MODEL_PATHS = {
    "external_a": EXTERNAL / "next_input_seed_1.txt",
    "external_b": EXTERNAL / "next_input_seed_2.txt",
    "remaining_a": REMAINING / "direct_seed_a.txt",
    "remaining_b": REMAINING / "direct_seed_b.txt",
    "domain": DOMAIN / "model_fit_A.txt",
}
INPUT_PATHS = {
    "protocol": PROTOCOL,
    "core_ab_calibration": CORE,
    "model_controls_ab_calibration": MODEL_CAL,
    "qfloor_development": QFLOOR,
    "real_model_dev_dry_run": DEV_DRY,
    "first_full_remaining_lock": BASE / "budget_calibration_lock.json",
    "feature_source": ROOT / "scripts/experiments/develop_pagarsky_context_exposure.py",
    "external_fit_script": ROOT / "scripts/experiments/fit_exgentic_next_input_baselines.py",
    "external_fit_receipt": EXTERNAL / "fit_receipt.json",
    "external_clock": EXTERNAL / "call_clock_input.json",
    "domain_fit_script": ROOT / "scripts/experiments/develop_liveclawbench_next_input_control.py",
    "domain_fit_lock": DOMAIN / "primary_Afit_Bcal_lock.json",
    **{f"model_{name}": path for name, path in MODEL_PATHS.items()},
}
EXPECTED = {
    "source": "580009458bb5691be0a41337a23dd4c66dc5595d5362e359be2430c6ef67126e",
    "identity": "07355c4a66a784a6fb019c261ca195a25d09e01380b0b8abf1c6840101ac8b7e",
    "core_ab_calibration": "d9187f020356ccf1c85279c260ec912059c9748817b7ec76a2dbaee7ee1d2e5c",
    "model_controls_ab_calibration": "7fb33efcd5cf0a52bcac6e7d6b863030df73fd630b79cb4f2af67f61ef5ed02d",
    "qfloor_development": "5ea23422396cd0308c1fe4c5b4ed05eb855efe4e11396cf7f8bf98d04c54b119",
    "real_model_dev_dry_run": "0122aead5bd0cce962f85a88d64c03821b8b87a80d1e8606a4c98c44da19d936",
    "first_full_remaining_lock": "fcfb839d9426d45b09079c7f7c0c72d0846fd734989c23c5b8da43c374faf28d",
    "feature_source": "56cd7b790cb68388ac5dde458123c78ae8a0a7efba222a9773c69af6b577b0dc",
    "external_fit_script": "dbc470e5c5d76cfda9d7a647436915f15a11773ce43a5bb4695693c975a0368b",
    "external_fit_receipt": "ae9a89da25d868342b0f14a0f25a1a23aad83005edc32327972533d76adf84b4",
    "external_clock": "ee87a18e2a3e492fb31c8861c778d828e1aa9e6d803dcd6286ea3f27be965d6e",
    "domain_fit_script": "92c4b76e8018b2590aaa82ff873fa64fd5df3d4d9e5ef61dde8769df1303f364",
    "domain_fit_lock": "2bcb08849acfde0c933329f1002d3f195aa02d8288abf1c0b312af5bfbea3e9f",
    "model_external_a": "fd7d6c0303f5aa2f322bc9d5c504618e0ae99b77486ec707ba3f8d0f826d1cb0",
    "model_external_b": "a9943a7c7ca430df643dcd3c4ba02c14ab60f258a229908bb732f4763fecf3ab",
    "model_remaining_a": "ac34b2c603b62c067d4ec756918984e4c197ce94012d9bf8e9f8b47a2dbd5006",
    "model_remaining_b": "d38e2d9dc789f9cf4613baf5c6365b9ceb320fc797b8acc66c659a1cc6969f9b",
    "model_domain": "e11b9d50b5e11a2343bf85834c70580c7ac6b0af8940eaba03bc61271213ddfe",
}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    if OUT.exists() or any((OUTDIR / name).exists() for name in (
        "predictions.jsonl", "prediction_manifest.json", "score_started.json", "score_report.json"
    )):
        raise FileExistsError("v2 output already exists")
    if digest(SOURCE) != EXPECTED["source"] or digest(MANIFEST) != EXPECTED["identity"]:
        raise ValueError("Pinned source or identity changed")
    for name, path in INPUT_PATHS.items():
        if digest(path) != EXPECTED.get(name, digest(path)):
            raise ValueError(f"Pretest input changed: {name}")
    core = json.loads(CORE.read_text(encoding="utf-8"))
    model_cal = json.loads(MODEL_CAL.read_text(encoding="utf-8"))
    identity = json.loads(MANIFEST.read_text(encoding="utf-8"))
    domain_lock = json.loads((DOMAIN / "primary_Afit_Bcal_lock.json").read_text(encoding="utf-8"))
    if len(identity["test_case_ids"]) != 90 or len(core["active_actors"]) != 14:
        raise ValueError("Cohort or actor count changed")
    if set(core["A_case_ids"]) & set(core["B_case_ids"]):
        raise ValueError("A/B overlap")
    if domain_lock["actor_codes"].keys() != core["actor_caps_from_A"].keys():
        raise ValueError("Domain actor encoding differs")
    model_q = model_cal["q_raw_and_final"]
    if model_q["0.5"]["historical"]["q_final"] != 0.0:
        raise ValueError("Primary median threshold must be zero")
    if model_q["0.5"]["historical"]["q_raw"] != core["primary_decision_risk_from_B"]["0.5"]["q"]:
        raise ValueError("Primary raw threshold mismatch")
    if any(values["q_final"] != max(0.0, values["q_raw"])
           for per_cap in model_q.values() for values in per_cap.values()):
        raise ValueError("q floor mismatch")
    lock = {
        "status": "frozen pretest; 90-case trajectories and scores unopened",
        "source_revision": "c3cfa229a90a8d540c88b9c490ac3dd88db1425a",
        "source_sha256": EXPECTED["source"],
        "identity_sha256": EXPECTED["identity"],
        "test_case_ids": identity["test_case_ids"],
        "test_case_ids_sha256": identity["test_case_ids_sha256"],
        "planned_test_slots": 4590,
        "A_case_ids": core["A_case_ids"],
        "B_case_ids": core["B_case_ids"],
        "active_actors": core["active_actors"],
        "actor_eligible_A": core["actor_eligible_A"],
        "actor_caps_A": core["actor_caps_from_A"],
        "model_decision_q_raw": {
            cap: {arm: values["q_raw"] for arm, values in arms.items()}
            for cap, arms in model_q.items()
        },
        "model_decision_q_final": {
            cap: {arm: values["q_final"] for arm, values in arms.items()}
            for cap, arms in model_q.items()
        },
        "actor_input_drop_q_B": core["actor_input_drop_q_from_B"],
        "joint_input_drop_q_B": core["joint_input_drop_q_from_B"],
        "global_input_drop_q_B": core["raw_global_input_drop_q_from_B"],
        "full_remaining_q_B": core["full_remaining_q_from_B"],
        "full_remaining_q_old": core["full_remaining_q_from_first_44_lock"],
        "domain_actor_codes": domain_lock["actor_codes"],
        "model_paths": {name: relative(path) for name, path in MODEL_PATHS.items()},
        "clock_path": relative(EXTERNAL / "call_clock_input.json"),
        "runner_sha256": digest(RUNNER),
        "lock_writer_sha256": digest(Path(__file__)),
        "inputs": {
            name: {"relative_path": relative(path), "sha256": digest(path)}
            for name, path in sorted(INPUT_PATHS.items())
        },
        "primary_cap_quantile": "0.5",
        "primary_threshold_floor": 0.0,
        "primary_min_pooled_saving_bootstrap_lower": 0.01,
        "quality_noninferiority_margin": -0.01,
        "learned_control_noninferiority_margin": -0.005,
        "case_bootstrap_draws": 10000,
        "case_bootstrap_seed": 20260923,
        "test_trajectory_values_returned_to_python": 0,
        "test_score_values_returned_to_python": 0,
        "selection_chronology": "All 44 development outcomes and raw negative-q failures were inspected before the q>=0 amendment; the 90 test cases were unscored and unopened when this lock was written. No exact finite-sample conformal claim.",
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(lock, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({
        "lock": str(OUT), "sha256": digest(OUT),
        "runner_sha256": lock["runner_sha256"],
        "protocol_sha256": lock["inputs"]["protocol"]["sha256"],
        "test_case_count": len(lock["test_case_ids"]),
        "input_count": len(lock["inputs"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
