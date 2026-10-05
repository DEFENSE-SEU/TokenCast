from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "workspace/experiments/exgentic_v2_context_exposure_prefit_20260923_v2"
SESSIONS = OUTPUT / "test_per_session_scored.jsonl"
FAMILIES = OUTPUT / "test_per_family_scored.jsonl"
REPORT = OUTPUT / "test_score_report.json"
RECEIPT = OUTPUT / "test_score_receipt.json"
AUDIT = OUTPUT / "test_independent_arithmetic_audit.json"
METHODS = ("context_exposure", "count_mean", "direct_two_seed", "call_clock")
REFERENCES = ("count_mean", "direct_two_seed", "call_clock")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _family_rows(sessions: list[dict]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for row in sessions:
        grouped[row["family_key"]].append(row)
    result = {}
    for family, rows in sorted(grouped.items()):
        assert len({row["benchmark"] for row in rows}) == 1
        result[family] = {
            "benchmark": rows[0]["benchmark"],
            "sessions": len(rows),
            "target": sum(row["mean_target"] for row in rows) / len(rows),
            "error": {
                method: sum(row["mean_abs_error"][method] for row in rows) / len(rows)
                for method in METHODS
            },
        }
    return result


def _wape(rows: list[dict]) -> dict[str, float]:
    denominator = sum(row["target"] for row in rows)
    if denominator <= 0:
        raise ValueError("nonpositive family target denominator")
    return {method: sum(row["error"][method] for row in rows) / denominator
            for method in METHODS}


def _ci(rows: list[dict], seed: int) -> dict:
    target = np.asarray([row["target"] for row in rows], dtype=np.float64)
    context = np.asarray([row["error"]["context_exposure"] for row in rows], dtype=np.float64)
    direct = np.asarray([row["error"]["direct_two_seed"] for row in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(rows), size=(5000, len(rows)))
    delta = ((context[draws] - direct[draws]).sum(axis=1)
             / target[draws].sum(axis=1))
    return {
        "families": len(rows),
        "context_minus_direct": _wape(rows)["context_exposure"] - _wape(rows)["direct_two_seed"],
        "paired_family_bootstrap_ci95": [float(value) for value in np.quantile(delta, [0.025, 0.975])],
    }


def _strata(sessions: list[dict], dimension: str) -> dict:
    grouped = defaultdict(list)
    for row in sessions:
        grouped[str(row[dimension])].append(row)
    return {
        key: {"sessions": len(rows), **_ci(list(_family_rows(rows).values()), 20260924)}
        for key, rows in sorted(grouped.items())
    }


def main() -> int:
    if AUDIT.exists():
        raise ValueError("independent arithmetic audit exists; refusing overwrite")
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    for name, path in (("per_session_sha256", SESSIONS), ("per_family_sha256", FAMILIES),
                       ("report_sha256", REPORT)):
        if receipt[name] != _hash(path):
            raise ValueError(f"scored artifact hash mismatch: {path}")
    sessions = [json.loads(line) for line in SESSIONS.read_text(encoding="utf-8").splitlines()]
    recorded_families = {
        row["family_key"]: row
        for row in (json.loads(line) for line in FAMILIES.read_text(encoding="utf-8").splitlines())
    }
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    families = _family_rows(sessions)
    if len(sessions) != 1338 or len(families) != 277 or set(families) != set(recorded_families):
        raise ValueError("scored session/family coverage mismatch")
    if sum(row["prefixes"] for row in sessions) != 31764:
        raise ValueError("scored prefix count mismatch")
    for key, calculated in families.items():
        recorded = recorded_families[key]
        if calculated["sessions"] != recorded["sessions"] or calculated["benchmark"] != recorded["benchmark"]:
            raise ValueError("family identity/count mismatch")
        if not np.isclose(calculated["target"], recorded["mean_target"], rtol=0, atol=1e-9):
            raise ValueError("family target arithmetic mismatch")
        for method in METHODS:
            if not np.isclose(calculated["error"][method], recorded["mean_abs_error"][method],
                              rtol=0, atol=1e-9):
                raise ValueError("family error arithmetic mismatch")
    by_benchmark = defaultdict(list)
    for row in families.values():
        by_benchmark[row["benchmark"]].append(row)
    if len(by_benchmark) != 6:
        raise ValueError("benchmark coverage mismatch")
    benchmark_wape = {bench: _wape(rows) for bench, rows in sorted(by_benchmark.items())}
    macro = {method: sum(values[method] for values in benchmark_wape.values()) / 6
             for method in METHODS}
    pooled = _wape(list(families.values()))
    recorded = report["metrics"]
    for method in METHODS:
        if not np.isclose(macro[method], recorded["six_benchmark_macro_wape"][method], rtol=0, atol=1e-12):
            raise ValueError("macro WAPE arithmetic mismatch")
        if not np.isclose(pooled[method], recorded["pooled_family_equal_wape"][method], rtol=0, atol=1e-12):
            raise ValueError("pooled WAPE arithmetic mismatch")
        for bench, values in benchmark_wape.items():
            if not np.isclose(values[method], recorded["benchmark"][bench]["wape"][method],
                              rtol=0, atol=1e-12):
                raise ValueError("benchmark WAPE arithmetic mismatch")
    rng = np.random.default_rng(20260923)
    macro_samples = {method: np.zeros(5000, dtype=np.float64) for method in METHODS}
    for bench in sorted(by_benchmark):
        rows = by_benchmark[bench]
        target = np.asarray([row["target"] for row in rows])
        draws = rng.integers(0, len(rows), size=(5000, len(rows)))
        denominator = target[draws].sum(axis=1)
        for method in METHODS:
            error = np.asarray([row["error"][method] for row in rows])
            macro_samples[method] += error[draws].sum(axis=1) / denominator / 6
    comparison_ci = {}
    for reference in REFERENCES:
        delta = macro_samples["context_exposure"] - macro_samples[reference]
        ci = [float(value) for value in np.quantile(delta, [0.025, 0.975])]
        recorded_ci = recorded["comparisons"][reference]["paired_family_bootstrap_ci95"]
        if not np.allclose(ci, recorded_ci, rtol=0, atol=1e-12):
            raise ValueError("primary bootstrap CI mismatch")
        comparison_ci[reference] = ci
    audit = {
        "source_score_receipt_sha256": _hash(RECEIPT),
        "scored_sessions": len(sessions), "scored_families": len(families),
        "scored_prefixes": 31764,
        "independent_family_session_benchmark_macro_pooled_arithmetic_match": True,
        "independent_primary_bootstrap_ci_match": True,
        "six_benchmark_macro_wape": macro,
        "pooled_family_equal_wape": pooled,
        "primary_comparison_ci95": comparison_ci,
        "benchmark_context_minus_direct": {
            bench: _ci(rows, 20260924) for bench, rows in sorted(by_benchmark.items())
        },
        "actor_context_minus_direct": _strata(sessions, "model"),
        "status_context_minus_direct": _strata(sessions, "status"),
        "harness_context_minus_direct": _strata(sessions, "harness"),
    }
    with AUDIT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(audit, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"independent_arithmetic_match": True,
                      "primary_macro": macro,
                      "test_families": len(families),
                      "audit_sha256": _hash(AUDIT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
