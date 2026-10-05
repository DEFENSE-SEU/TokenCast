from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

from scripts.experiments.develop_pagarsky_context_exposure import FEATURES, _features
from scripts.experiments.fit_exgentic_v2_context_exposure import (
    COLUMNS, FEATURE_SOURCE, HEADS, LOCK, MANIFEST, METHODS, OUTPUT,
    RECEIPT, ROOT, ROSTER, SOURCE, _load_development, _load_roster, _weights,
    _weighted_median,
)
from scripts.experiments.run_public_trajectory_validation import _sha256


PREDICTIONS = OUTPUT / "test_predictions.jsonl"
PREDICTION_MANIFEST = OUTPUT / "test_prediction_manifest.json"
CLOCK_FILE = OUTPUT / "train_dev_call_clock.json"
FIT_SCRIPT = ROOT / "scripts/experiments/fit_exgentic_v2_context_exposure.py"


def _load_models(fit_receipt: dict) -> dict[str, lgb.Booster]:
    models = {}
    for head in HEADS:
        source = OUTPUT / "models" / f"{head}.txt"
        if _sha256(source) != fit_receipt["model_sha256"][head]:
            raise ValueError(f"fitted model hash changed: {head}")
        models[head] = lgb.Booster(model_file=str(source))
    return models


def _clock_lookup(roster: dict[str, dict]) -> dict[int, float]:

    traces = _load_development(roster)
    rows = list(roster.values())
    selected = {session for session, row in roster.items() if row["split"] != "test"}
    weights = _weights(rows, selected)
    values = defaultdict(list)
    for session in selected:
        inputs, outputs = traces[session]
        suffix = [0] * (len(inputs) + 1)
        for j in range(len(inputs) - 1, -1, -1):
            suffix[j] = suffix[j + 1] + inputs[j] + outputs[j]
        for k in range(1, len(inputs)):
            values[k].append((suffix[k], weights[session]))
    return {
        int(k): _weighted_median(
            np.asarray([item[0] for item in pairs], dtype=np.float64),
            np.asarray([item[1] for item in pairs], dtype=np.float64),
        )
        for k, pairs in sorted(values.items())
    }


def _load_test(roster: dict[str, dict]) -> dict[str, tuple[list[int], list[int]]]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    if lock["revision"] != receipt["revision"] or len(lock["files"]) != 9:
        raise ValueError("source lock differs from receipt")
    expected = {session for session, row in roster.items() if row["split"] == "test"}
    traces = {}
    for entry, verified in zip(lock["files"], receipt["files"]):
        if entry["sha256"] != verified["sha256"] or entry["path"] != verified["path"]:
            raise ValueError("source receipt mismatch")
        source = SOURCE / entry["path"]
        if source.stat().st_size != entry["size"] or _sha256(source) != entry["sha256"]:
            raise ValueError(f"source file changed: {source}")
        reader = pq.ParquetFile(source)
        for batch in reader.iter_batches(batch_size=64, columns=COLUMNS):
            for row in batch.to_pylist():
                session = row["session_id"]
                if session not in expected:
                    continue
                if session in traces:
                    raise ValueError("duplicate test session")
                spans = row["spans"] or []
                if len(spans) != roster[session]["observed_successful_calls"]:
                    raise ValueError("test span count differs from roster")
                stamps = [span["start_time"] for span in spans]
                if any(stamp is None for stamp in stamps) or stamps != sorted(stamps):
                    raise ValueError("nonchronological test trace")
                if any(span["status"]["code"] != 1 for span in spans):
                    raise ValueError("failed span in test trace")
                inputs = [span["attributes"]["gen_ai.usage.input_tokens"] for span in spans]
                outputs = [span["attributes"]["gen_ai.usage.output_tokens"] for span in spans]
                if any(type(i) is not int or type(o) is not int or i < 0 or o < 0 or i + o <= 0
                       for i, o in zip(inputs, outputs)):
                    raise ValueError("invalid test usage")
                traces[session] = (inputs, outputs)
    if set(traces) != expected:
        raise ValueError(f"test traces missing: {len(expected - set(traces))}")
    return traces


def main() -> int:
    if PREDICTIONS.exists() or PREDICTION_MANIFEST.exists() or CLOCK_FILE.exists():
        raise ValueError("test prediction artifact exists; refusing overwrite")
    fit_receipt = json.loads((OUTPUT / "fit_receipt.json").read_text(encoding="utf-8"))
    roster, split_manifest = _load_roster()
    expected_hashes = {
        "source_lock_sha256": _sha256(LOCK),
        "download_receipt_sha256": _sha256(RECEIPT),
        "roster_sha256": _sha256(ROSTER),
        "split_manifest_sha256": _sha256(MANIFEST),
        "feature_source_sha256": _sha256(FEATURE_SOURCE),
        "fit_script_sha256": _sha256(FIT_SCRIPT),
    }
    if any(fit_receipt[key] != value for key, value in expected_hashes.items()):
        raise ValueError("fit receipt source/code lock changed")
    if _sha256(OUTPUT / "development_report.json") != fit_receipt["development_report_sha256"]:
        raise ValueError("development report changed")
    models = _load_models(fit_receipt)
    clock = _clock_lookup(roster)
    with CLOCK_FILE.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump({str(k): value for k, value in sorted(clock.items())}, stream, indent=2)
        stream.write("\n")
    traces = _load_test(roster)
    sessions = sorted(traces)
    x, row_keys = [], []
    for session in sessions:
        inputs, outputs = traces[session]
        for k in range(1, len(inputs) + 1):
            x.append(_features(inputs[:k], outputs[:k]))
            row_keys.append((session, k))
    matrix = np.asarray(x, dtype=np.float64)
    if matrix.shape != (33102, len(FEATURES)) or not np.all(np.isfinite(matrix)):
        raise ValueError("test feature matrix invalid")
    heads = {head: model.predict(matrix) for head, model in models.items()}
    current = matrix[:, FEATURES.index("current_input")]
    n = np.maximum(1.0, heads["future_calls"])
    results = {
        "context_exposure": np.maximum(0.0, n * (current + heads["mean_growth"]
                                                  + np.maximum(0.0, heads["mean_output"]))),
        "count_mean": n * np.maximum(0.0, heads["mean_total"]),
        "direct_two_seed": np.maximum(0.0, (heads["direct_seed_a"] + heads["direct_seed_b"]) / 2),
        "call_clock": np.asarray([
            clock[min(clock, key=lambda known: abs(known - k))] for _session, k in row_keys
        ]),
    }
    if tuple(results) != METHODS or any(np.any(~np.isfinite(values)) or np.any(values < 0)
                                        for values in results.values()):
        raise ValueError("invalid test predictions")
    terminal = Counter()
    by_benchmark = Counter()
    with PREDICTIONS.open("x", encoding="utf-8", newline="\n") as stream:
        for index, (session, k) in enumerate(row_keys):
            source = roster[session]
            is_terminal = k == source["observed_successful_calls"]
            terminal[is_terminal] += 1
            by_benchmark[source["benchmark"]] += 1
            row = {
                "session_id": session, "family_key": source["family_key"],
                "benchmark": source["benchmark"], "model": source["model"],
                "harness": source["harness"], "k": k, "terminal": is_terminal,
                "predictions": {name: float(values[index]) for name, values in results.items()},
            }
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    if terminal != {False: 31764, True: 1338}:
        raise ValueError("test terminal/nonterminal count changed")
    manifest = {
        "source_revision": split_manifest["source_revision"],
        **expected_hashes,
        "fit_receipt_sha256": _sha256(OUTPUT / "fit_receipt.json"),
        "prediction_script_sha256": _sha256(Path(__file__)),
        "model_sha256": fit_receipt["model_sha256"],
        "clock_lookup_sha256": _sha256(CLOCK_FILE),
        "prediction_sha256": _sha256(PREDICTIONS),
        "test_sessions": len(sessions), "test_families": 277,
        "prediction_rows": len(row_keys), "terminal_rows": terminal[True],
        "nonterminal_rows": terminal[False],
        "rows_by_benchmark": dict(by_benchmark),
        "methods": METHODS, "feature_names": FEATURES,
        "test_suffix_targets_computed": False,
        "test_scores_computed": False,
    }
    with PREDICTION_MANIFEST.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"prediction_sha256": manifest["prediction_sha256"],
                      "prediction_rows": len(row_keys), "terminal_rows": terminal[True]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
