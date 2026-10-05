from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

from scripts.experiments.develop_pagarsky_context_exposure import FEATURES, _features, _params
from scripts.experiments.run_public_trajectory_validation import _sha256


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "workspace/experiments/exgentic_v2_feasibility_20260923"
SOURCE = ROOT / "workspace/external/exgentic_agent_llm_traces_v2_4b8ad4a"
ROSTER = AUDIT / "clean_task_family_split_v2.jsonl"
MANIFEST = AUDIT / "clean_task_family_split_v2_manifest.json"
LOCK = AUDIT / "source_lock.json"
RECEIPT = AUDIT / "download_receipt.json"
FEATURE_SOURCE = ROOT / "scripts/experiments/develop_pagarsky_context_exposure.py"
PROTOCOL = ROOT / "docs/local/exgentic-v2-context-exposure-prefit-protocol-2026-09-23.md"
OUTPUT = ROOT / "workspace/experiments/exgentic_v2_context_exposure_prefit_20260923_v1"
FEATURE_SHA = "56cd7b790cb68388ac5dde458123c78ae8a0a7efba222a9773c69af6b577b0dc"
ROSTER_SHA = "fa72c9a09ab20054214c2b507d816b74988dfbd4409b40b351e833356587b033"
HEADS = {
    "future_calls": ("future_calls", 240, 20260923, False),
    "mean_growth": ("mean_growth", 240, 20260923, False),
    "mean_output": ("mean_output", 240, 20260923, False),
    "mean_total": ("mean_total", 240, 20260923, False),
    "direct_seed_a": ("target", 360, 20260923, True),
    "direct_seed_b": ("target", 360, 20270923, True),
}
METHODS = ("context_exposure", "count_mean", "direct_two_seed", "call_clock")
COLUMNS = [
    "session_id", "spans.list.element.start_time", "spans.list.element.status.code",
    "spans.list.element.attributes.gen_ai.usage.input_tokens",
    "spans.list.element.attributes.gen_ai.usage.output_tokens",
]


def _load_roster() -> tuple[dict[str, dict], dict]:
    if _sha256(FEATURE_SOURCE) != FEATURE_SHA or _sha256(ROSTER) != ROSTER_SHA:
        raise ValueError("feature source or roster changed after prefit lock")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["roster_sha256"] != ROSTER_SHA or manifest["family_intersection_across_splits"] != 0:
        raise ValueError("split manifest changed")
    roster = {}
    for line in ROSTER.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["session_id"] in roster:
            raise ValueError("duplicate session ID")
        roster[row["session_id"]] = row
    if len(roster) != 6764 or Counter(r["split"] for r in roster.values()) != {
        "train": 4033, "dev": 1393, "test": 1338,
    }:
        raise ValueError("split roster counts changed")
    return roster, manifest


def _load_development(roster: dict[str, dict]) -> dict[str, tuple[list[int], list[int]]]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    if receipt["revision"] != lock["revision"] or len(receipt["files"]) != 9:
        raise ValueError("source receipt incomplete")
    for a, b in zip(lock["files"], receipt["files"]):
        if a["path"] != b["path"] or a["sha256"] != b["sha256"]:
            raise ValueError("source receipt differs from lock")
    expected = {session for session, row in roster.items() if row["split"] != "test"}
    traces = {}
    for entry in lock["files"]:
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
                    raise ValueError("duplicate selected trace")
                spans = row["spans"] or []
                if len(spans) != roster[session]["observed_successful_calls"]:
                    raise ValueError("span count differs from roster")
                stamps = [span["start_time"] for span in spans]
                if any(stamp is None for stamp in stamps) or stamps != sorted(stamps):
                    raise ValueError("nonchronological selected trace")
                if any(span["status"]["code"] != 1 for span in spans):
                    raise ValueError("failed span in selected trace")
                inputs = [span["attributes"]["gen_ai.usage.input_tokens"] for span in spans]
                outputs = [span["attributes"]["gen_ai.usage.output_tokens"] for span in spans]
                if any(type(i) is not int or type(o) is not int or i < 0 or o < 0 or i + o <= 0
                       for i, o in zip(inputs, outputs)):
                    raise ValueError("invalid selected usage")
                traces[session] = (inputs, outputs)
    if set(traces) != expected:
        raise ValueError(f"selected development traces missing: {len(expected - set(traces))}")
    return traces


def _weights(rows: list[dict], selected: set[str]) -> dict[str, float]:
    family_sessions = Counter(row["family_key"] for row in rows if row["session_id"] in selected)
    benchmark_families = defaultdict(set)
    for row in rows:
        if row["session_id"] in selected:
            benchmark_families[row["benchmark"]].add(row["family_key"])
    if len(benchmark_families) != 6:
        raise ValueError("missing benchmark in training weights")
    return {
        row["session_id"]: 1.0 / (
            6 * len(benchmark_families[row["benchmark"]])
            * family_sessions[row["family_key"]] * row["nonterminal_prefixes"]
        )
        for row in rows if row["session_id"] in selected
    }


def _points(rows: list[dict], traces: dict[str, tuple[list[int], list[int]]],
            selected: set[str]) -> dict:
    per_prefix_weights = _weights(rows, selected)
    x, y, n, growth, future_output, mean_total = [], [], [], [], [], []
    sessions, families, benchmarks, ks, weights = [], [], [], [], []
    for row in rows:
        session = row["session_id"]
        if session not in selected:
            continue
        inputs, outputs = traces[session]
        m = len(inputs)
        suffix_input = [0] * (m + 1)
        suffix_output = [0] * (m + 1)
        for index in range(m - 1, -1, -1):
            suffix_input[index] = suffix_input[index + 1] + inputs[index]
            suffix_output[index] = suffix_output[index + 1] + outputs[index]
        for k in range(1, m):
            remaining = m - k
            future_i = suffix_input[k]
            future_o = suffix_output[k]
            x.append(_features(inputs[:k], outputs[:k]))
            y.append(float(future_i + future_o))
            n.append(float(remaining))
            growth.append(future_i / remaining - inputs[k - 1])
            future_output.append(future_o / remaining)
            mean_total.append((future_i + future_o) / remaining)
            sessions.append(session)
            families.append(row["family_key"])
            benchmarks.append(row["benchmark"])
            ks.append(k)
            weights.append(per_prefix_weights[session])
    result = {
        "x": np.asarray(x, dtype=np.float64),
        "target": np.asarray(y, dtype=np.float64),
        "future_calls": np.asarray(n, dtype=np.float64),
        "mean_growth": np.asarray(growth, dtype=np.float64),
        "mean_output": np.asarray(future_output, dtype=np.float64),
        "mean_total": np.asarray(mean_total, dtype=np.float64),
        "session": sessions, "family": families, "benchmark": benchmarks,
        "k": np.asarray(ks, dtype=np.int32),
        "weight": np.asarray(weights, dtype=np.float64),
    }
    if result["x"].shape != (len(x), len(FEATURES)) or not np.all(np.isfinite(result["x"])):
        raise ValueError("invalid feature matrix")
    if not np.isclose(result["weight"].sum(), 1.0):
        raise ValueError("training weights do not sum to one")
    return result


def _dataset(points: dict, label: str) -> lgb.Dataset:
    return lgb.Dataset(points["x"], label=points[label], weight=points["weight"],
                       feature_name=list(FEATURES), free_raw_data=False)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    return float(values[order][np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2.0)])


def _clock(train: dict, ks: np.ndarray) -> np.ndarray:
    by_k = {}
    for k in np.unique(train["k"]):
        match = train["k"] == k
        by_k[int(k)] = _weighted_median(train["target"][match], train["weight"][match])
    return np.asarray([by_k[min(by_k, key=lambda known: abs(known - int(k)))] for k in ks])


def _predict(points: dict, models: dict[str, lgb.Booster], clock: np.ndarray) -> dict[str, np.ndarray]:
    heads = {name: model.predict(points["x"]) for name, model in models.items()}
    n = np.maximum(1.0, heads["future_calls"])
    current = points["x"][:, FEATURES.index("current_input")]
    result = {
        "context_exposure": np.maximum(0.0, n * (current + heads["mean_growth"]
                                                  + np.maximum(0.0, heads["mean_output"]))),
        "count_mean": n * np.maximum(0.0, heads["mean_total"]),
        "direct_two_seed": np.maximum(0.0, (heads["direct_seed_a"] + heads["direct_seed_b"]) / 2),
        "call_clock": clock,
    }
    if any(len(values) != len(points["k"]) or not np.all(np.isfinite(values))
           or np.any(values < 0) for values in result.values()):
        raise ValueError("invalid development predictions")
    return result


def _family_summary(points: dict, predictions: dict[str, np.ndarray]) -> dict[str, dict]:
    session_indices = defaultdict(list)
    for index, session in enumerate(points["session"]):
        session_indices[session].append(index)
    family_sessions = defaultdict(list)
    for session, indices in session_indices.items():
        family = points["family"][indices[0]]
        benchmark = points["benchmark"][indices[0]]
        family_sessions[family].append({
            "benchmark": benchmark,
            "target": float(np.mean(points["target"][indices])),
            "error": {name: float(np.mean(np.abs(values[indices] - points["target"][indices])))
                      for name, values in predictions.items()},
        })
    return {
        family: {
            "benchmark": sessions[0]["benchmark"],
            "sessions": len(sessions),
            "target": float(np.mean([row["target"] for row in sessions])),
            "error": {name: float(np.mean([row["error"][name] for row in sessions]))
                      for name in METHODS},
        }
        for family, sessions in family_sessions.items()
    }


def _metrics(families: dict[str, dict]) -> dict:
    by_benchmark = defaultdict(list)
    for row in families.values():
        by_benchmark[row["benchmark"]].append(row)
    if len(by_benchmark) != 6:
        raise ValueError("missing benchmark in development summary")
    benchmark_results = {
        bench: {
            "families": len(rows), "sessions": sum(row["sessions"] for row in rows),
            "wape": {name: sum(row["error"][name] for row in rows)
                     / sum(row["target"] for row in rows) for name in METHODS},
            "family_equal_mae": {name: float(np.mean([row["error"][name] for row in rows]))
                                 for name in METHODS},
        }
        for bench, rows in sorted(by_benchmark.items())
    }
    macro = {name: float(np.mean([result["wape"][name] for result in benchmark_results.values()]))
             for name in METHODS}
    all_rows = list(families.values())
    pooled = {name: sum(row["error"][name] for row in all_rows)
              / sum(row["target"] for row in all_rows) for name in METHODS}
    rng = np.random.default_rng(20260923)
    bootstrap = {name: np.zeros(5000, dtype=np.float64) for name in METHODS}
    for rows in by_benchmark.values():
        target = np.asarray([row["target"] for row in rows])
        draws = rng.integers(0, len(rows), size=(5000, len(rows)))
        denominator = target[draws].sum(axis=1)
        for name in METHODS:
            error = np.asarray([row["error"][name] for row in rows])
            bootstrap[name] += error[draws].sum(axis=1) / denominator / 6
    comparisons = {
        reference: {
            "delta_context_minus_reference": macro["context_exposure"] - macro[reference],
            "paired_family_bootstrap_ci95": [float(q) for q in np.quantile(
                bootstrap["context_exposure"] - bootstrap[reference], [0.025, 0.975])],
        }
        for reference in ("count_mean", "direct_two_seed", "call_clock")
    }
    return {"benchmark": benchmark_results, "six_benchmark_macro_wape": macro,
            "pooled_family_equal_wape": pooled, "comparisons": comparisons,
            "bootstrap_replicates": 5000}


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("prefit output exists; refusing overwrite")
    roster, split_manifest = _load_roster()
    rows = list(roster.values())
    traces = _load_development(roster)
    train_sessions = {r["session_id"] for r in rows if r["split"] == "train"}
    dev_sessions = {r["session_id"] for r in rows if r["split"] == "dev"}
    train = _points(rows, traces, train_sessions)
    dev = _points(rows, traces, dev_sessions)
    if len(train["x"]) != 96009 or len(dev["x"]) != 29972:
        raise ValueError("development prefix counts differ from locked roster")
    if _features([1], [2]) != _features([1, 1000][:1], [2, 2000][:1]):
        raise ValueError("fixture prefix causality check failed")
    dev_models = {}
    best_rounds = {}
    for head, (label, max_rounds, seed, stochastic) in HEADS.items():
        model = lgb.train(
            _params(seed, stochastic), _dataset(train, label),
            num_boost_round=max_rounds, valid_sets=[_dataset(dev, label)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        best_rounds[head] = int(model.best_iteration)
        if not 1 <= best_rounds[head] <= max_rounds:
            raise ValueError(f"invalid best round: {head}")
        dev_models[head] = model
        print(json.dumps({"development_head": head, "best_round": best_rounds[head]}), flush=True)
    dev_predictions = _predict(dev, dev_models, _clock(train, dev["k"]))
    dev_metrics = _metrics(_family_summary(dev, dev_predictions))
    combined_sessions = train_sessions | dev_sessions
    combined = {
        name: np.concatenate([train[name], dev[name]]) if isinstance(train[name], np.ndarray)
        else train[name] + dev[name]
        for name in train
    }
    combined_weights = _weights(rows, combined_sessions)
    combined["weight"] = np.asarray([combined_weights[session]
                                     for session in combined["session"]], dtype=np.float64)
    if not np.isclose(combined["weight"].sum(), 1.0):
        raise ValueError("refit weights do not sum to one")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    models_dir = OUTPUT / "models"
    models_dir.mkdir()
    model_hashes = {}
    for head, (label, _max_rounds, seed, stochastic) in HEADS.items():
        model = lgb.train(_params(seed, stochastic), _dataset(combined, label),
                          num_boost_round=best_rounds[head])
        destination = models_dir / f"{head}.txt"
        model.save_model(str(destination))
        model_hashes[head] = _sha256(destination)
        print(json.dumps({"refit_head": head, "sha256": model_hashes[head]}), flush=True)
    dev_report = {
        "source_revision": split_manifest["source_revision"],
        "train_sessions": len(train_sessions), "train_prefixes": len(train["x"]),
        "dev_sessions": len(dev_sessions), "dev_prefixes": len(dev["x"]),
        "best_rounds": best_rounds, "metrics": dev_metrics,
        "test_rows_projected_and_skipped_during_source_scan": True,
        "test_suffix_targets_computed": False,
    }
    report_file = OUTPUT / "development_report.json"
    with report_file.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dev_report, stream, indent=2)
        stream.write("\n")
    receipt = {
        "source_revision": split_manifest["source_revision"],
        "source_lock_sha256": _sha256(LOCK),
        "download_receipt_sha256": _sha256(RECEIPT),
        "roster_sha256": _sha256(ROSTER),
        "split_manifest_sha256": _sha256(MANIFEST),
        "feature_source_sha256": _sha256(FEATURE_SOURCE),
        "prefit_protocol_sha256": _sha256(PROTOCOL),
        "fit_script_sha256": _sha256(Path(__file__)),
        "feature_names": FEATURES,
        "head_specifications": HEADS,
        "best_rounds": best_rounds,
        "model_sha256": model_hashes,
        "development_report_sha256": _sha256(report_file),
        "train_and_dev_families": 1100,
        "test_families": 277,
        "test_suffix_targets_read": False,
        "test_model_selection_performed": False,
        "paid_api_calls": 0,
    }
    with (OUTPUT / "fit_receipt.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"six_benchmark_macro_wape": dev_metrics["six_benchmark_macro_wape"],
                      "comparisons": dev_metrics["comparisons"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
