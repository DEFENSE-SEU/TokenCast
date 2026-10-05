from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCORED = ROOT / "workspace/experiments/liveclawbench_budget_v2_20260923"
OUTDIR = ROOT / "workspace/experiments/liveclawbench_budget_v2_independent_audit_20260923"
OUT = OUTDIR / "audit.json"
EXPECTED_REPORT_SHA256 = "054749d5dc0a298c20711727d5e2f511ec1ec2fb1cc29058c88d871e9a8b35d2"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-11 * max(1.0, abs(a), abs(b))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    report_path = SCORED / "score_report.json"
    if sha(report_path) != EXPECTED_REPORT_SHA256:
        raise ValueError("One-shot score report changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cases = jsonl(SCORED / "per_case_scored.jsonl")
    slots = jsonl(SCORED / "per_slot_scored.jsonl")
    manifest = json.loads((SCORED / "prediction_manifest.json").read_text(encoding="utf-8"))
    score_started = json.loads((SCORED / "score_started.json").read_text(encoding="utf-8"))
    if len(cases) != 90 or len(slots) != 4590 or len({s["sample_id"] for s in slots}) != 4590:
        raise ValueError("Case/slot denominator changed")
    if sha(SCORED / "per_case_scored.jsonl") != report["per_case_sha256"]:
        raise ValueError("Per-case checksum differs from score report")
    if sha(SCORED / "per_slot_scored.jsonl") != report["per_slot_sha256"]:
        raise ValueError("Per-slot checksum differs from score report")
    if sha(SCORED / "predictions.jsonl") != manifest["predictions_sha256"]:
        raise ValueError("Prediction checksum differs from manifest")
    if score_started["predictions_sha256"] != manifest["predictions_sha256"]:
        raise ValueError("Score receipt predates different prediction file")
    if manifest["score_column_projected"] is not False or score_started["score_column_projected_before_receipt"] is not False:
        raise ValueError("Score projection chronology flag changed")
    strict = [slot for slot in slots if slot["strict_scored"]]
    if len(strict) != report["strict_active_traces"]:
        raise ValueError("Strict trace count mismatch")
    models = sorted(report["cap_grid"]["0.5"]["arms"])
    arithmetic = {}
    for p in ("0.25", "0.5", "0.75"):
        reference = sum(case["grid"][p]["reference_cost_tokens"] for case in cases)
        if reference != report["cap_grid"][p]["reference_cost_tokens"]:
            raise ValueError("Reference cost mismatch")
        by_model = {}
        for model in models:
            saved = sum(case["grid"][p]["arms"][model]["saved_tokens"] for case in cases)
            alerts = sum(case["grid"][p]["arms"][model]["alerts"] for case in cases)
            false = sum(case["grid"][p]["arms"][model]["false_aborts"] for case in cases)
            given = report["cap_grid"][p]["arms"][model]
            if (saved, alerts, false) != (given["saved_tokens"], given["alerts"], given["false_aborts"]):
                raise ValueError(f"Arm arithmetic mismatch: {p}/{model}")
            if not close(saved / reference, given["pooled_saving_ratio"]):
                raise ValueError(f"Arm saving ratio mismatch: {p}/{model}")
            by_model[model] = {"saved_tokens": saved, "alerts": alerts,
                               "false_aborts": false, "pooled_ratio": saved / reference}
        arithmetic[p] = {"reference_cost_tokens": reference, "arms": by_model}
    for slot in strict:
        for p in ("0.25", "0.5", "0.75"):
            item = slot["grid"][p]
            ref = item["reference_cost_tokens"]
            for arm, result in item["arms"].items():
                saved = result["saved_tokens"]
                stop = result["stop"]
                if not (0 <= saved <= ref):
                    raise ValueError("Invalid per-slot saving")
                if stop is not None and saved != ref - stop["spent"]:
                    raise ValueError("Per-slot settled-cost mismatch")
                if stop is None and saved != 0:
                    raise ValueError("Unstopped slot has saving")
                if result["false_abort"] and (stop is None or item["reference_closed_ending_proxy"] != 1):
                    raise ValueError("False-abort definition mismatch")
    ref_case = np.asarray([case["grid"]["0.5"]["reference_cost_tokens"] for case in cases], dtype=np.float64)
    primary = np.asarray([case["grid"]["0.5"]["arms"]["primary_historical"]["saved_tokens"] for case in cases], dtype=np.float64)
    oracle = np.asarray([case["grid"]["0.5"]["arms"]["privileged_one_step_input"]["saved_tokens"] for case in cases], dtype=np.float64)
    external = np.asarray([case["grid"]["0.5"]["arms"]["external_direct"]["saved_tokens"] for case in cases], dtype=np.float64)
    domain = np.asarray([case["grid"]["0.5"]["arms"]["domain_direct"]["saved_tokens"] for case in cases], dtype=np.float64)
    rng = np.random.default_rng(20260923)
    indices = rng.integers(0, 90, size=(10000, 90))
    draws_ref = ref_case[indices].sum(axis=1)
    draws_primary = primary[indices].sum(axis=1) / draws_ref
    draws_ext_delta = (primary - external)[indices].sum(axis=1) / draws_ref
    draws_domain_delta = (primary - domain)[indices].sum(axis=1) / draws_ref
    draws_oracle_ratio = primary[indices].sum(axis=1) / oracle[indices].sum(axis=1)
    given = report["bootstrap"]
    checks = {
        "primary_pooled_saving_one_sided_lower": float(np.quantile(draws_primary, .05)),
        "primary_minus_external_one_sided_lower": float(np.quantile(draws_ext_delta, .05)),
        "primary_minus_domain_one_sided_lower": float(np.quantile(draws_domain_delta, .05)),
        "primary_to_privileged_one_step_ratio_one_sided_lower": float(np.quantile(draws_oracle_ratio, .05)),
    }
    expected = {
        "primary_pooled_saving_one_sided_lower": given["pooled_saving_ratio"]["primary_historical"]["bootstrap_one_sided_lower_95"],
        "primary_minus_external_one_sided_lower": given["primary_minus_controls_pooled_ratio"]["external_direct"]["bootstrap_one_sided_lower_95"],
        "primary_minus_domain_one_sided_lower": given["primary_minus_controls_pooled_ratio"]["domain_direct"]["bootstrap_one_sided_lower_95"],
        "primary_to_privileged_one_step_ratio_one_sided_lower": given["primary_divided_by_privileged_one_step_saving"]["bootstrap_one_sided_lower_95"],
    }
    if any(not close(checks[name], expected[name]) for name in checks):
        raise ValueError("Independent case-bootstrap mismatch")
    evaluable = [case for case in cases if case["strict_active_count"] > 0]
    qrng = np.random.default_rng(20260923)
    qi = qrng.integers(0, len(evaluable), size=(10000, len(evaluable)))
    quality = {}
    for key, ref_key in (("closed_ending_proxy", "reference_closed_ending_proxy"),
                         ("verifier_reward", "reference_verifier_reward")):
        differences = np.asarray([
            (case["grid"]["0.5"]["arms"]["primary_historical"][key]
             - case["grid"]["0.5"][ref_key]) / case["strict_active_count"]
            for case in evaluable
        ], dtype=np.float64)
        lower = float(np.quantile(differences[qi].mean(axis=1), .05))
        if not close(lower, given["quality_case_macro_difference"][key]["bootstrap_one_sided_lower_95"]):
            raise ValueError("Independent quality bootstrap mismatch")
        quality[key] = {"point": float(differences.mean()), "one_sided_lower": lower}
    completers = [case for case in cases if case["grid"]["0.5"]["reference_completing_traces"] > 0]
    no_false = sum(case["grid"]["0.5"]["arms"]["primary_historical"]["false_aborts"] == 0 for case in completers)
    if (len(completers), no_false) != (
        report["primary_false_abort_case_coverage"]["reference_completer_cases"],
        report["primary_false_abort_case_coverage"]["without_false_abort_among_reference_completer_cases"],
    ):
        raise ValueError("False-abort family denominator mismatch")
    result = {
        "status": "read-only independent arithmetic audit; no rescore or refit",
        "score_report_sha256": EXPECTED_REPORT_SHA256,
        "score_started_sha256": sha(SCORED / "score_started.json"),
        "prediction_manifest_sha256": sha(SCORED / "prediction_manifest.json"),
        "per_case_sha256": sha(SCORED / "per_case_scored.jsonl"),
        "per_slot_sha256": sha(SCORED / "per_slot_scored.jsonl"),
        "script_sha256": sha(Path(__file__)),
        "planned_slots": len(slots), "strict_active_traces": len(strict),
        "quality_evaluable_cases": len(evaluable),
        "arithmetic": arithmetic,
        "bootstrap_one_sided_lower_recomputed": checks,
        "quality_case_macro_recomputed": quality,
        "reference_completer_cases": len(completers),
        "without_primary_false_abort_among_completer_cases": no_false,
        "all_checks_passed": True,
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps({"audit": str(OUT), "sha256": sha(OUT),
                      "all_checks_passed": True, "strict": len(strict)}, sort_keys=True))


if __name__ == "__main__":
    main()
