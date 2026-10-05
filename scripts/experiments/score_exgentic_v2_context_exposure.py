from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from scripts.experiments.fit_exgentic_v2_context_exposure_v2 import (
    LOCK, MANIFEST, METHODS, OUTPUT, RECEIPT, ROSTER, SOURCE, _metrics,
)
from scripts.experiments.predict_exgentic_v2_context_exposure_v2 import (
    PREDICTIONS, PREDICTION_MANIFEST,
)
from scripts.experiments.run_public_trajectory_validation import _sha256
from scripts.experiments.verify_exgentic_v2_test_predictions_v2 import main as verify_predictions


PREDICTION_MANIFEST_SHA = "e4a3b7a41fc63419faf965c733b13f05bf2012122fece38f876ae40e70dd20f2"
PREDICTIONS_SHA = "781ed2c0934320ae0ab0ce6f1a449925e9bdbad7b0191446307c00b8b35dd721"
STARTED = OUTPUT / "test_score_started.json"
PER_SESSION = OUTPUT / "test_per_session_scored.jsonl"
PER_FAMILY = OUTPUT / "test_per_family_scored.jsonl"
REPORT = OUTPUT / "test_score_report.json"
SCORE_RECEIPT = OUTPUT / "test_score_receipt.json"
COLUMNS = [
    "session_id", "status", "spans.list.element.status.code",
    "spans.list.element.attributes.gen_ai.usage.input_tokens",
    "spans.list.element.attributes.gen_ai.usage.output_tokens",
]


def _session_loss(inputs: list[int], outputs: list[int],
                  predictions: list[dict]) -> tuple[float, dict[str, float]]:
    m = len(inputs)
    if m < 2 or len(predictions) != m:
        raise ValueError("scored trace or predictions incomplete")
    suffix = [0] * (m + 1)
    for j in range(m - 1, -1, -1):
        suffix[j] = suffix[j + 1] + inputs[j] + outputs[j]
    targets = [float(suffix[k]) for k in range(1, m)]
    mean_target = sum(targets) / len(targets)
    errors = {
        name: sum(abs(predictions[k - 1]["predictions"][name] - targets[k - 1])
                  for k in range(1, m)) / (m - 1)
        for name in METHODS
    }
    return mean_target, errors


def _family_rows(session_rows: list[dict]) -> dict[str, dict]:
    by_family = defaultdict(list)
    for row in session_rows:
        by_family[row["family_key"]].append(row)
    return {
        family: {
            "family_key": family,
            "benchmark": rows[0]["benchmark"],
            "sessions": len(rows),
            "mean_target": sum(row["mean_target"] for row in rows) / len(rows),
            "mean_abs_error": {
                name: sum(row["mean_abs_error"][name] for row in rows) / len(rows)
                for name in METHODS
            },
        }
        for family, rows in sorted(by_family.items())
    }


def _stratum(session_rows: list[dict], field: str) -> dict[str, dict]:
    by_group = defaultdict(list)
    for row in session_rows:
        by_group[str(row[field])].append(row)
    output = {}
    for name, rows in sorted(by_group.items()):
        families = _family_rows(rows)
        target = sum(row["mean_target"] for row in families.values())
        by_bench = Counter(row["benchmark"] for row in families.values())
        output[name] = {
            "sessions": len(rows), "families": len(families),
            "benchmark_family_counts": dict(by_bench),
            "pooled_family_equal_wape": {
                method: sum(row["mean_abs_error"][method] for row in families.values()) / target
                for method in METHODS
            },
        }
    return output


def _fixture_test() -> None:
    predictions = [
        {"predictions": {name: 20.0 for name in METHODS}},
        {"predictions": {name: 0.0 for name in METHODS}},
    ]
    target, error = _session_loss([10, 20], [1, 2], predictions)
    assert target == 22.0 and all(value == 2.0 for value in error.values())
    sessions = [
        {"family_key": "b:a", "benchmark": "b", "mean_target": 22.0,
         "mean_abs_error": {name: 2.0 for name in METHODS}},
        {"family_key": "b:a", "benchmark": "b", "mean_target": 42.0,
         "mean_abs_error": {name: 6.0 for name in METHODS}},
        {"family_key": "b:c", "benchmark": "b", "mean_target": 50.0,
         "mean_abs_error": {name: 10.0 for name in METHODS}},
    ]
    families = _family_rows(sessions)
    assert families["b:a"]["sessions"] == 2
    assert families["b:a"]["mean_target"] == 32.0
    assert families["b:a"]["mean_abs_error"][METHODS[0]] == 4.0
    assert (families["b:a"]["mean_abs_error"][METHODS[0]]
            + families["b:c"]["mean_abs_error"][METHODS[0]]) / (
                families["b:a"]["mean_target"] + families["b:c"]["mean_target"]) == 14 / 82


def _load_predictions(roster: dict[str, dict]) -> dict[str, list[dict]]:
    predictions = defaultdict(list)
    for line in PREDICTIONS.open(encoding="utf-8"):
        row = json.loads(line)
        if roster[row["session_id"]]["split"] != "test":
            raise ValueError("non-test prediction row")
        predictions[row["session_id"]].append(row)
    return predictions


def _load_session_losses(roster: dict[str, dict], predictions: dict[str, list[dict]]) -> list[dict]:
    source_lock = json.loads(LOCK.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected = {session for session, row in roster.items() if row["split"] == "test"}
    expected_ids = pa.array(sorted(expected), type=pa.string())
    seen = set()
    losses = []
    for entry, verified in zip(source_lock["files"], receipt["files"]):
        if entry["path"] != verified["path"] or entry["sha256"] != verified["sha256"]:
            raise ValueError("source receipt mismatch")
        source = SOURCE / entry["path"]
        if source.stat().st_size != entry["size"] or _sha256(source) != entry["sha256"]:
            raise ValueError("source hash changed")
        reader = pq.ParquetFile(source)
        for batch in reader.iter_batches(batch_size=64, columns=COLUMNS):
            selected = batch.filter(pc.is_in(batch.column(0), value_set=expected_ids))
            for row in selected.to_pylist():
                session = row["session_id"]
                if session in seen:
                    raise ValueError("duplicate source test trace")
                seen.add(session)
                spans = row["spans"] or []
                if len(spans) != roster[session]["observed_successful_calls"]:
                    raise ValueError("span count differs from roster")
                if any(span["status"]["code"] != 1 for span in spans):
                    raise ValueError("non-success span in clean test trace")
                inputs = [span["attributes"]["gen_ai.usage.input_tokens"] for span in spans]
                outputs = [span["attributes"]["gen_ai.usage.output_tokens"] for span in spans]
                mean_target, errors = _session_loss(inputs, outputs, predictions[session])
                identity = roster[session]
                losses.append({
                    "session_id": session, "family_key": identity["family_key"],
                    "benchmark": identity["benchmark"], "model": identity["model"],
                    "harness": identity["harness"], "status": row["status"],
                    "prefixes": len(spans) - 1, "mean_target": mean_target,
                    "mean_abs_error": errors,
                })
    if seen != expected or len(losses) != 1338:
        raise ValueError("test session losses incomplete")
    return sorted(losses, key=lambda row: row["session_id"])


def _score() -> int:
    _fixture_test()
    if any(path.exists() for path in (STARTED, PER_SESSION, PER_FAMILY, REPORT, SCORE_RECEIPT)):
        raise ValueError("one-shot test score artifact already exists")
    if _sha256(PREDICTION_MANIFEST) != PREDICTION_MANIFEST_SHA:
        raise ValueError("prediction manifest changed after review")
    if _sha256(PREDICTIONS) != PREDICTIONS_SHA:
        raise ValueError("prediction file changed after review")
    verify_predictions()
    roster = {row["session_id"]: row for row in
              (json.loads(line) for line in ROSTER.read_text(encoding="utf-8").splitlines())}
    if _sha256(ROSTER) != json.loads(MANIFEST.read_text(encoding="utf-8"))["roster_sha256"]:
        raise ValueError("roster changed")
    predictions = _load_predictions(roster)
    with STARTED.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump({"prediction_manifest_sha256": PREDICTION_MANIFEST_SHA,
                   "predictions_sha256": PREDICTIONS_SHA,
                   "scorer_sha256": _sha256(Path(__file__)),
                   "test_score_started_once": True}, stream, indent=2)
        stream.write("\n")
    session_rows = _load_session_losses(roster, predictions)
    families = _family_rows(session_rows)
    if len(families) != 277 or sum(row["prefixes"] for row in session_rows) != 31764:
        raise ValueError("test family or prefix count changed")
    adapted = {
        family: {"benchmark": row["benchmark"], "sessions": row["sessions"],
                 "target": row["mean_target"], "error": row["mean_abs_error"]}
        for family, row in families.items()
    }
    metrics = _metrics(adapted)
    metrics["by_actor"] = _stratum(session_rows, "model")
    metrics["by_harness"] = _stratum(session_rows, "harness")
    metrics["by_top_level_status"] = _stratum(session_rows, "status")
    for path, rows in ((PER_SESSION, session_rows), (PER_FAMILY, list(families.values()))):
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "source_revision": json.loads(MANIFEST.read_text(encoding="utf-8"))["source_revision"],
        "test_families": len(families), "test_sessions": len(session_rows),
        "nonterminal_prefixes": 31764, "methods": METHODS,
        "metrics": metrics,
        "endpoint": "observed remaining successful-agent-chat input+output token volume",
        "no_test_driven_model_changes": True,
    }
    with REPORT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    with SCORE_RECEIPT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump({
            "score_started_sha256": _sha256(STARTED),
            "scorer_sha256": _sha256(Path(__file__)),
            "prediction_manifest_sha256": PREDICTION_MANIFEST_SHA,
            "predictions_sha256": PREDICTIONS_SHA,
            "per_session_sha256": _sha256(PER_SESSION),
            "per_family_sha256": _sha256(PER_FAMILY),
            "report_sha256": _sha256(REPORT),
        }, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"six_benchmark_macro_wape": metrics["six_benchmark_macro_wape"],
                      "comparisons": metrics["comparisons"]}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--self-test", action="store_true")
    choice.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _fixture_test()
        print(json.dumps({"fixture_arithmetic_and_family_weighting_passed": True,
                          "test_source_or_predictions_read": False}))
        return 0
    return _score()


if __name__ == "__main__":
    raise SystemExit(main())
