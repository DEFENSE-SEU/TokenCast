from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "workspace/experiments/envbench_four_actor_20260923"
OUTPUT = ROOT / "workspace/experiments/envbench_four_actor_repro_supplement_20260923"
EXPECTED_PROTOCOL = "f27842fd5cc403dfb2aefb4f1df65828542248f2cad84976d2958ecc9e119524"
EXPECTED_SCORE = "b95223a78a66956f4d0594064c8f56277ee5746b9f3b34a6fc3abcad375b4728"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def copy_into(source: Path, relative: str) -> None:
    destination = OUTPUT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def write_cases(score: dict) -> None:
    arms = ("certificate", "previous_input", "learned_raw", "learned_minus64", "residual_raw", "residual_minus64")
    columns = [
        "actor", "task_id", "terminal_class", "total_observed_tokens", "reference_tokens",
        "cost_known", "original_observed_at_or_under_cap", "eligible_points",
        "engineering_lower_bound_violations", "strict_verifier_success", "verifier_status",
    ]
    for arm in arms:
        columns.extend((f"{arm}_alert", f"{arm}_saved_tokens", f"{arm}_false_abort", f"{arm}_original_success_marked"))
    with (OUTPUT / "per_case_outcomes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for case in sorted(score["cases"], key=lambda item: (item["task_id"], item["actor"])):
            row = {key: case.get(key) for key in columns if key in case}
            for arm in arms:
                value = case["arms"][arm]
                row[f"{arm}_alert"] = value["alert"]
                row[f"{arm}_saved_tokens"] = value["saved_tokens"]
                row[f"{arm}_false_abort"] = value["false_abort_closed_undercap"]
                row[f"{arm}_original_success_marked"] = value["original_verifier_success_interrupted"]
            writer.writerow(row)


def write_repositories(score: dict) -> None:
    by_id: dict[str, list[dict]] = {}
    for case in score["cases"]:
        by_id.setdefault(case["task_id"], []).append(case)
    assert len(by_id) == 132 and all(len(rows) == 4 for rows in by_id.values())
    columns = [
        "task_id", "actor_count", "identified_actor_count", "reference_tokens_sum",
        "guard_saved_tokens_sum", "previous_saved_tokens_sum", "learned_raw_saved_tokens_sum",
        "residual_raw_saved_tokens_sum", "closed_undercap_actor_count",
        "guard_false_abort_count", "original_strict_success_count",
        "guard_original_success_marked_count",
    ]
    with (OUTPUT / "per_repository_outcomes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for task_id, cases in sorted(by_id.items()):
            writer.writerow({
                "task_id": task_id,
                "actor_count": len(cases),
                "identified_actor_count": sum(c["cost_known"] for c in cases),
                "reference_tokens_sum": sum(c["reference_tokens"] for c in cases),
                "guard_saved_tokens_sum": sum(c["arms"]["certificate"]["saved_tokens"] for c in cases),
                "previous_saved_tokens_sum": sum(c["arms"]["previous_input"]["saved_tokens"] for c in cases),
                "learned_raw_saved_tokens_sum": sum(c["arms"]["learned_raw"]["saved_tokens"] for c in cases),
                "residual_raw_saved_tokens_sum": sum(c["arms"]["residual_raw"]["saved_tokens"] for c in cases),
                "closed_undercap_actor_count": sum(c["captured_completion"] for c in cases),
                "guard_false_abort_count": sum(c["arms"]["certificate"]["false_abort_closed_undercap"] for c in cases),
                "original_strict_success_count": sum(c["strict_verifier_success"] for c in cases),
                "guard_original_success_marked_count": sum(c["arms"]["certificate"]["original_verifier_success_interrupted"] for c in cases),
            })


def main() -> None:
    protocol_path = EXPERIMENT / "protocol.lock.json"
    score_path = EXPERIMENT / "score.json"
    assert digest(protocol_path) == EXPECTED_PROTOCOL
    assert digest(score_path) == EXPECTED_SCORE
    protocol = load_json(protocol_path)
    score = load_json(score_path)
    assert score["status"] == "one_shot_scored"
    assert score["protocol_sha256"] == EXPECTED_PROTOCOL
    assert len(score["cases"]) == 528
    assert not OUTPUT.exists(), f"Refusing to overwrite {OUTPUT}"
    OUTPUT.mkdir(parents=True)

    experiment_files = (
        "protocol.lock.json", "protocol.lock.pretest-v1.json",
        "pretest_review_receipt.json", "pretest_review_receipt.pretest-v1.json",
        "pretest_failure_v1.json", "prediction_started.json", "prediction_receipt.json",
        "predictions.jsonl", "raw_test_source_receipt.json", "result_source_receipt.json",
        "score_started.json", "score.json",
    )
    for name in experiment_files:
        copy_into(EXPERIMENT / name, f"frozen/{name}")

    identity_path = ROOT / "workspace/experiments/envbench_budget_feasibility_20260923/jvm_bash_agent_4o_cohort.lock.json"
    assert digest(identity_path) == protocol["identity_lock_sha256"]
    copy_into(identity_path, "cohort/jvm_bash_agent_4o_cohort.lock.json")
    copy_into(ROOT / "workspace/experiments/envbench_budget_feasibility_20260923/evaluation_code_pin.json", "cohort/evaluation_code_pin.json")

    code_paths = {
        "predictor": "scripts/experiments/predict_envbench_four_actor_once.py",
        "scorer": "scripts/experiments/score_envbench_four_actor_once.py",
        "fetcher": "scripts/experiments/fetch_envbench_four_actor_test_sources.py",
        "results_fetcher": "scripts/experiments/fetch_envbench_four_actor_test_results.py",
        "freezer": "scripts/experiments/freeze_envbench_four_actor_protocol.py",
    }
    for role, source in code_paths.items():
        path = ROOT / source
        assert digest(path) == protocol["code_sha256"][role], role
        copy_into(path, f"code/{path.name}")
    for source in (
        "scripts/experiments/develop_envbench_residual_on_transport.py",
        "workspace/experiments/envbench_budget_independent_review_20260923/learned_cross_actor_delta_tooltoken22_reproduce.py",
        "workspace/experiments/envbench_budget_independent_review_20260923/audit_four_actor_postscore.py",
    ):
        path = ROOT / source
        copy_into(path, f"code/{path.name}")

    for actor, settings in protocol["actors"].items():
        for group in ("learned_model_artifacts", "residual_model_artifacts"):
            for item in settings[group]:
                path = ROOT / item["path"]
                assert digest(path) == item["sha256"], item["path"]
                copy_into(path, f"models/{actor}/{path.name}")

    for source, destination in (
        ("workspace/experiments/envbench_budget_independent_review_20260923/four_actor_postscore_independent_audit.md", "audit/independent_postscore_audit.md"),
        ("docs/local/envbench-four-actor-postscore-quality-audit-2026-09-23.md", "audit/quality_audit.md"),
        ("docs/local/envbench-four-actor-one-shot-result-2026-09-23.md", "audit/one_shot_result.md"),
        ("workspace/experiments/envbench_cross_actor_development_20260923/residual_on_transport_v1/report.json", "development/residual_on_transport_report.json"),
    ):
        copy_into(ROOT / source, destination)

    write_cases(score)
    write_repositories(score)

    readme = """# EnvBench four-actor local reproducibility supplement

**Status: local review package only; no public release has occurred.** This package contains no raw trajectory or result-source blobs, prompts, tool outputs, or credentials. Its model files and numerical predictions are derived from the pinned public source. Do not describe the automated audits as human review.

Source: https://huggingface.co/datasets/JetBrains-Research/EnvBench-trajectories at revision `4ffd02bed0f39fd87bffe3be7edf2440aba16cd1`. Published evaluator source was separately pinned to Git commit `1ba2695018080df3a3606b100fc1dbfad96ea115`; its present code may differ from the exporter that produced the traces. Official benchmark: https://github.com/JetBrains-Research/EnvBench .

The immutable cohort lock assigns 400 fit, 132 calibration, and 132 test `repo@commit` identities; the same 132 test identities appear under four JVM agent/model configurations. One Installamatic-mini fit trajectory is absent. The 132 test identities and all four source-path/blob-ID maps are also in `frozen/protocol.lock.json`. The four actor B50 caps, gate, 64-token empirical margin, model hashes, scoring definition, and five success criteria were selected using fit/calibration and fixed before test-event parsing.

The V1 fetcher stopped during its own code-hash check before downloading or parsing test bytes (`frozen/pretest_failure_v1.json`). V1 protocol and pretest review were archived. V2 corrected that file-byte hash, then froze protocol and received a fresh automated pretest review. V2 hashed 528 test trajectories, sealed 10,659 numerical prediction rows before reading result rows, then verified four result-source files and scored once. The source and result file receipts are included; the files themselves remain in the pinned public dataset.

`frozen/` contains both protocol versions, review/failure receipts, source/result hash receipts, sealed numerical predictions, prediction/score start receipts, and the one-shot score. `code/` contains the exact pinned predictor, scorer, fetchers and freezer, plus development/audit scripts. `models/` contains the 16 frozen LightGBM models. `per_case_outcomes.csv` and `per_repository_outcomes.csv` contain no source text; the latter sums four actor outcomes by repository identity for concentration checks. The primary savings metric first pools saved/reference tokens *within each actor*, then averages the four percentages equally. The repository CSV's raw sums are an audit aid and are not that estimand.

`manifest.json` records source paths, sizes and SHA-256 digests for every payload file. `SHA256SUMS.txt` additionally covers `manifest.json`. Validate the package with a SHA-256 checker, then compare the protocol and score hashes to the values in `manifest.json`. To fully rerun predictions, fetch the original fit/test trajectories from the pinned public dataset and preserve their verified blobs. The local package alone does not establish an online intervention, true task correctness, or monetary savings.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    payload = sorted(path for path in OUTPUT.rglob("*") if path.is_file())
    files = [{"path": path.relative_to(OUTPUT).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)} for path in payload]
    manifest = {
        "status": "local_review_package_not_publicly_released",
        "source_dataset": "JetBrains-Research/EnvBench-trajectories",
        "source_revision": protocol["source_revision"],
        "published_evaluator_commit": "1ba2695018080df3a3606b100fc1dbfad96ea115",
        "protocol_sha256": EXPECTED_PROTOCOL,
        "score_sha256": EXPECTED_SCORE,
        "identity_lock_sha256": protocol["identity_lock_sha256"],
        "code_sha256": protocol["code_sha256"],
        "fit_calibration_test_identity_counts": [400, 132, 132],
        "test_actor_count": 4,
        "test_task_actor_count": 528,
        "files": files,
    }
    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUTPUT / "SHA256SUMS.txt").open("w", encoding="utf-8") as stream:
        for path in sorted([*payload, manifest_path]):
            stream.write(f"{digest(path)}  {path.relative_to(OUTPUT).as_posix()}\n")
    print(json.dumps({"package": str(OUTPUT), "files": len(files) + 2, "bytes": sum(path.stat().st_size for path in OUTPUT.rglob("*") if path.is_file())}, indent=2))


if __name__ == "__main__":
    main()
