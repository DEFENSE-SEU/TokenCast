from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "workspace/experiments/envbench_python_transfer_20260923"
OUTPUT = ROOT / "workspace/experiments/envbench_python_transfer_repro_supplement_20260923"
PROTOCOL_SHA = "b3ef5067ab9a7e91a1d702882b17621b12e1eb9a80017bc57528e4c5e3280748"
SCORE_SHA = "b3342d8d7599483a97c8e9bc70f8fa64960f84185d155074fb4b04d5634fd3e5"
RECOVERY_SHA = "823425366b5db823009edf026ea010c7d9ca3c8f498a5fa1bcc936f1e5c75d0e"
ORIGINS: dict[str, str] = {}


def digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def copy(source: Path, relative: str, expected: str | None = None) -> None:
    assert source.is_file(), source
    if expected is not None:
        assert digest(source) == expected, source
    destination = OUTPUT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    assert digest(destination) == digest(source)
    ORIGINS[relative] = source.relative_to(ROOT).as_posix()


def write_cases(score: dict) -> None:
    arms = tuple(score["metrics"]["arms"])
    fields = [
        "task_id", "source_sha256", "terminal_class", "total_observed_tokens",
        "cost_known", "reference_tokens", "captured_completion",
        "strict_verifier_success", "verifier_status", "real_calls",
        "non_request_rows", "cache_read_tokens", "eligible_points",
        "eligible_input_tokens", "transport_absolute_input_error",
        "previous_absolute_input_error", "learned_absolute_input_error",
        "residual_absolute_input_error", "engineering_lower_bound_violations",
        "certificate_hit_points",
    ]
    for arm in arms:
        fields += [f"{arm}_{name}" for name in (
            "alert", "stop_call_index", "spent_at_stop", "saved_tokens",
            "false_abort_closed_undercap", "original_verifier_success_interrupted",
            "retained_observable_verifier_success")]
    with (OUTPUT / "per_case_outcomes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in score["cases"]:
            row = {name: case[name] for name in fields if name in case}
            row["non_request_rows"] = case.get("non_request_rows", case.get("synthetic_non_request_rows"))
            for arm in arms:
                for name, value in case["arms"][arm].items():
                    row[f"{arm}_{name}"] = value
            writer.writerow(row)


def main() -> None:
    assert not OUTPUT.exists(), f"Refusing to overwrite {OUTPUT}"
    protocol_path = EXPERIMENT / "protocol.lock.json"
    score_path = EXPERIMENT / "score.json"
    recovery_path = EXPERIMENT / "recovery_protocol.lock.json"
    assert digest(protocol_path) == PROTOCOL_SHA
    assert digest(score_path) == SCORE_SHA
    assert digest(recovery_path) == RECOVERY_SHA
    protocol, score, recovery = load(protocol_path), load(score_path), load(recovery_path)
    prediction_receipt = load(EXPERIMENT / "prediction_receipt.json")
    postscore_audit = load(EXPERIMENT / "postscore_arithmetic_audit.json")
    assert len(protocol["test_ids"]) == len(score["cases"]) == 327
    assert score["protocol_sha256"] == prediction_receipt["protocol_sha256"] == PROTOCOL_SHA
    assert score["result_sha256"] == recovery["result_sha256"]
    assert postscore_audit["score_sha256"] == SCORE_SHA
    assert postscore_audit["status"] == "independent_postscore_arithmetic_passed"
    assert prediction_receipt["total_rows"] == 3131
    assert score["metrics"]["point_eligible_calls"] == 2457
    assert score["metrics"]["pre_registered_checks"]["mechanism_confirmation"] is True
    assert score["metrics"]["pre_registered_checks"]["strong_control_confirmation"] is False

    OUTPUT.mkdir(parents=True)
    frozen_names = (
        "protocol.lock.pretest-v1.json", "protocol.lock.pretest-v2.json",
        "protocol.lock.pretest-v3.json", "protocol.lock.pretest-v4.json",
        "protocol.lock.json", "pretest_review_receipt.json",
        "prediction_started.json", "prediction_receipt.json", "predictions.jsonl",
        "score_started.json", "recovery_protocol.lock.json",
        "recovery_pretest_review_receipt.json", "recovery_score_started.json",
        "score.json", "postscore_arithmetic_audit.json",
    )
    for name in frozen_names:
        copy(EXPERIMENT / name, f"frozen/{name}")
    copy(ROOT / "workspace/experiments/envbench_python_second_source_20260923/source_feasibility.json",
         "source/source_feasibility.json", protocol["source_audit_sha256"])
    copy(ROOT / "workspace/experiments/envbench_four_actor_20260923/protocol.lock.json",
         "source/jvm_four_actor_protocol.lock.json", protocol["jvm_protocol_sha256"])

    code = {
        "predictor": "scripts/experiments/predict_envbench_python_transfer_once.py",
        "scorer": "scripts/experiments/score_envbench_python_transfer_once.py",
        "source_feasibility_audit": "scripts/experiments/audit_envbench_python_second_source.py",
        "recovery": "scripts/experiments/recover_envbench_python_transfer_score_once.py",
        "jvm_predictor": "scripts/experiments/predict_envbench_four_actor_once.py",
        "jvm_scorer": "scripts/experiments/score_envbench_four_actor_once.py",
        "transitive_features": "scripts/experiments/develop_envbench_jvm_budget_models.py",
        "freezer": "scripts/experiments/freeze_envbench_python_transfer_protocol.py",
        "postscore_auditor": "scripts/experiments/audit_envbench_python_transfer_postscore.py",
        "independent_raw_auditor": "scripts/experiments/audit_envbench_python_transfer_final.py",
        "package_builder": "scripts/experiments/build_envbench_python_transfer_repro_package.py",
    }
    expected = {
        "predictor": protocol["code_sha256"]["predictor"],
        "scorer": protocol["code_sha256"]["scorer"],
        "source_feasibility_audit": protocol["code_sha256"]["source_feasibility_audit"],
        "recovery": recovery["recovery_script_sha256"],
        "jvm_predictor": protocol["imported_jvm_code_sha256"]["predictor"],
        "jvm_scorer": protocol["imported_jvm_code_sha256"]["scorer"],
        "transitive_features": protocol["transitive_feature_code_sha256"],
    }
    code_hashes = {}
    for role, relative in code.items():
        path = ROOT / relative
        copy(path, f"code/{path.name}", expected.get(role))
        code_hashes[role] = digest(path)
    model_hashes = {}
    for family, items in (("learned", protocol["learned_model_artifacts"]),
                          ("residual", protocol["residual_model_artifacts"])):
        for item in items:
            path = ROOT / item["path"]
            copy(path, f"models/{family}/{path.name}", item["sha256"])
            model_hashes[item["path"]] = item["sha256"]
    docs = (
        "docs/local/envbench-python-second-source-feasibility-2026-09-23.md",
        "docs/local/envbench-python-transfer-preregistered-protocol-2026-09-23.md",
        "docs/local/envbench-python-transfer-postscore-audit-2026-09-23.md",
    )
    for relative in docs:
        path = ROOT / relative
        copy(path, f"notes/{path.name}")

    audit_run = subprocess.run(
        [sys.executable, str(ROOT / code["independent_raw_auditor"])],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    independent = json.loads(audit_run.stdout)
    assert independent["status"] == "pass" and independent["score_sha256"] == SCORE_SHA
    (OUTPUT / "audit/independent_raw_audit.json").parent.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "audit/independent_raw_audit.json").write_text(
        json.dumps(independent, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_cases(score)

    readme = f"""# EnvBench Python transfer: local reproducibility supplement

**Status: local review package; no public release has occurred.** This package has no raw trajectory events, result rows, tool outputs, prompts, container logs, or credentials. It contains numerical predictions, numerical case outcomes, source metadata and hashes, frozen models, code, and automated audit receipts. The audited raw files remain outside this package.

The public source is [JetBrains-Research/EnvBench-trajectories](https://huggingface.co/datasets/JetBrains-Research/EnvBench-trajectories) at revision `{protocol['source_revision']}`. The published evaluator code was pinned to commit `{protocol['source_code_commit']}`. `source/source_feasibility.json` holds source IDs, sizes, blob IDs, and the excluded schema probe metadata. The roster has 327 Python Bash-agent test identities after excluding that one probe. The Python test identities do not overlap the JVM training/evaluation identities by exact ID.

## Frozen chronology

1. The excluded first result row and its matching trajectory were used to establish the Python schema. The source-feasibility receipt says the 327 test trajectories and result rows were unopened before the final lock. Archived protocol versions V1–V4 show pretest revisions; `frozen/protocol.lock.json` is the final approved lock, SHA-256 `{PROTOCOL_SHA}`. `frozen/pretest_review_receipt.json` records an automated independent pretest check with zero test events or result rows parsed.
2. The prediction run exclusively started, verified and parsed 327 pinned trajectories, then sealed 3,131 numerical rows in `frozen/predictions.jsonl`. Its receipt says no nonprobe result row had been opened. The final prediction SHA-256 is `{prediction_receipt['predictions_sha256']}`.
3. The first scoring process wrote `frozen/score_started.json` and stopped during a slow raw result download, before parsing result rows. The interrupted partial file was 10,485,760 bytes. The recovery lock pinned the full 182,813,874-byte raw object with SHA-256 `{recovery['result_sha256']}`. `frozen/recovery_pretest_review_receipt.json` records another automated independent pretest review. The exclusive recovery start preceded the one-time result parse and final score.
4. The final score SHA-256 is `{SCORE_SHA}`. `frozen/postscore_arithmetic_audit.json` and `audit/independent_raw_audit.json` are separate automated postscore checks. Both read the final source/results after the sealed score. `audit/independent_raw_audit.json` verifies 327 source SHA/blob IDs, all 3,131 prediction prefixes and gates, all arm first stops, result identity joins, strict Pyright quality fields, and the paired bootstrap.

## Result and interpretation

At B=47,078, 2,457 eligible calls provide 11,230,736 true input tokens. Point WAPE is 0.003419% for transported input, 11.901785% for previous input, 0.481865% for the learned delta, and 0.010480% for residual-on-transport. The empirical input lower estimate has zero observed violations among these eligible calls. That finite-sample check does not establish a universal lower bound.

The certificate alerts on 93 of 327 cases and saves 353,209 of 11,772,200 identifiable reference tokens (3.000365%). Repository-paired bootstrap gives a one-sided 95% lower saving of 2.524727% and a certificate-minus-previous-input lower difference of +0.070790 percentage points. Certificate, learned-minus-64, and residual-minus-64 produce identical first-stop savings on this test; the preregistered mechanism check passes and the strong-control check fails. The result supports this specific frozen transfer mechanism and does not establish superiority over the strong same-information controls.

There are 324 model-stop-closed traces and three step-limit-censored traces. All 327 cost references are identifiable at this cap. Among 230 closed under-cap traces, the certificate has zero observed false aborts and retains all 15 strict Pyright-proxy successes. The exact one-sided 95% upper rate for closed under-cap false aborts is 1.2940%. Five original strict successes over cap have interrupted trajectories under the certificate; their counterfactual quality after stopping is unobserved. Of 327 result lines, 326 match the frozen test IDs: one expected ID is absent and the excluded probe ID appears in the file. A separate zero-exit result has invalid Pyright output, leaving quality unknown for two cases.

`per_case_outcomes.csv` contains numeric outcomes and classifications for all 327 test identities. `manifest.json` records SHA-256, size, and source path for every payload file; `SHA256SUMS.txt` also covers the manifest. To rerun raw parsing, retrieve the pinned public source bytes and verify the result SHA plus each trajectory blob ID against `frozen/protocol.lock.json`. This local supplement alone cannot reproduce raw parsing because it deliberately excludes those bytes. The offline first-stop replay does not measure an online intervention, task correctness, or monetary savings.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    payload = sorted(path for path in OUTPUT.rglob("*") if path.is_file())
    files = []
    for path in payload:
        relative = path.relative_to(OUTPUT).as_posix()
        files.append({"path": relative, "bytes": path.stat().st_size,
                      "sha256": digest(path),
                      "origin_workspace_path": ORIGINS.get(relative, "generated within package")})
    manifest = {
        "status": "local_review_package_not_publicly_released",
        "source_dataset": protocol["repo"],
        "source_revision": protocol["source_revision"],
        "published_evaluator_commit": protocol["source_code_commit"],
        "final_protocol_sha256": PROTOCOL_SHA,
        "final_score_sha256": SCORE_SHA,
        "recovery_protocol_sha256": RECOVERY_SHA,
        "pinned_raw_result_sha256_excluded_from_package": recovery["result_sha256"],
        "test_id_count": len(protocol["test_ids"]),
        "prediction_row_count": prediction_receipt["total_rows"],
        "code_sha256": code_hashes,
        "model_sha256": model_hashes,
        "files": files,
    }
    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    with (OUTPUT / "SHA256SUMS.txt").open("w", encoding="utf-8") as stream:
        for path in sorted([*payload, manifest_path]):
            stream.write(f"{digest(path)}  {path.relative_to(OUTPUT).as_posix()}\n")
    print(json.dumps({"package": str(OUTPUT), "payload_files": len(payload),
                      "total_files": len(payload) + 2,
                      "bytes": sum(path.stat().st_size for path in OUTPUT.rglob("*")
                                   if path.is_file())}, indent=2))


if __name__ == "__main__":
    main()
