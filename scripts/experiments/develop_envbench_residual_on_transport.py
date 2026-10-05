from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import tiktoken


ROOT = Path(__file__).resolve().parents[2]
INDEPENDENT = ROOT / "workspace/experiments/envbench_budget_independent_review_20260923"
sys.path.insert(0, str(INDEPENDENT))
import learned_cross_actor_delta_tooltoken22_reproduce as prior

OUT = ROOT / "workspace/experiments/envbench_cross_actor_development_20260923/residual_on_transport_v1"
MODELS = OUT / "models"
POINTS = OUT / "calibration_points.jsonl"
REPORT = OUT / "report.json"
SEEDS = (20260923, 20270923)
FEATURE_NAMES = tuple(prior.FEATURE_NAMES) + ("transport_base",)
assert len(FEATURE_NAMES) == 23


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def verified_json(path: Path, expected: str) -> dict:
    if digest(path) != expected:
        raise ValueError(f"Development source drift: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def points_for(run: str, group: str, lock: dict, tree: dict, encoder) -> list[dict]:
    ids = lock["fit_ids"] if group == "fit" else lock["calibration_ids"]
    metadata = (lock["trajectory_metadata"] if run == "jvm/bash_agent-4o"
                else tree["runs"][run]["trajectory_metadata"])
    excluded = set(lock["untouched_test_ids"]) | {lock["schema_probe_excluded_id"]}
    if not set(ids).isdisjoint(excluded):
        raise ValueError("Held-out or excluded identity entered development extraction")
    output = []
    for task_id in ids:
        if task_id not in metadata:
            if group == "fit" and run != "jvm/bash_agent-4o":
                continue
            raise ValueError(f"Missing source metadata: {run} {group} {task_id}")
        path = ((prior.ORIGINAL / f"raw_{group}" / f"{task_id}.jsonl")
                if run == "jvm/bash_agent-4o"
                else prior.folder(run, group) / f"{task_id}.jsonl")
        rows = prior.extract(path, metadata[task_id], task_id, group, encoder,
                             prior.FIT_MEDIAN_OFFSET[run])
        for row in rows:
            raw_base = row["transported_input"] - prior.FIT_MEDIAN_OFFSET[run]
            if raw_base < 0 or len(row["features"]) != 22:
                raise ValueError("Transport base or feature schema invalid")
            row["transport_base_raw"] = raw_base
            row["features"] = row["features"] + [float(raw_base)]
            row["residual_target"] = row["observed_input"] - raw_base
            output.append(row)
    return output


def train(run: str, fit: list[dict], seed: int) -> tuple[lgb.Booster, dict]:
    matrix = np.asarray([p["features"] for p in fit], dtype=np.float64)
    target = np.asarray([p["residual_target"] for p in fit], dtype=np.float64)
    counts = Counter(p["task_id"] for p in fit)
    weight = np.asarray([1 / counts[p["task_id"]] for p in fit], dtype=np.float64)
    dataset = lgb.Dataset(matrix, label=target, weight=weight,
                          feature_name=list(FEATURE_NAMES))
    params = {"objective": "regression_l1", "learning_rate": 0.05,
              "num_leaves": 15, "min_data_in_leaf": 40,
              "feature_fraction": 0.9, "bagging_fraction": 0.9,
              "bagging_freq": 1, "lambda_l2": 1.0,
              "verbosity": -1, "num_threads": 1, "seed": seed,
              "feature_fraction_seed": seed, "bagging_seed": seed}
    model = lgb.train(params, dataset, num_boost_round=180)
    if tuple(model.feature_name()) != FEATURE_NAMES:
        raise ValueError("Stored model feature order drift")
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / f"{run.replace('/', '_')}_residual_seed_{seed}.txt"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(model.model_to_string())
    meta = {"path": path.relative_to(ROOT).as_posix(),
            "sha256": digest(path), "seed": seed,
            "feature_names": model.feature_name()}
    return model, meta


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    err = np.abs(actual - predicted)
    return {"mae": float(err.mean()), "wape": float(err.sum() / actual.sum()),
            "p95_abs_error": float(np.quantile(err, 0.95, method="nearest")),
            "max_abs_error": float(err.max())}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Development control already exists: {OUT}")
    lock = verified_json(prior.LOCK_PATH, prior.LOCK_SHA)
    tree = verified_json(prior.TREE, prior.TREE_SHA)
    dev = verified_json(prior.DEVELOPMENT, prior.DEVELOPMENT_SHA)
    cert = verified_json(prior.CERTIFICATE, prior.CERTIFICATE_SHA)
    if lock["revision"] != tree["revision"] or len(lock["untouched_test_ids"]) != 132:
        raise ValueError("Frozen source or split drift")
    if tiktoken.__version__ != "0.14.0":
        raise ValueError("Tokenizer package drift")
    encoder = tiktoken.get_encoding("o200k_base")
    OUT.mkdir(parents=True)
    summary = {}
    all_rows = []
    for run in prior.ACTORS:
        fit = points_for(run, "fit", lock, tree, encoder)
        cal = points_for(run, "calibration", lock, tree, encoder)
        expected_fit = (cert["actors"][run]["fit_gate_points"]
                        if run == "jvm/bash_agent-4o"
                        else dev["runs"][run]["fit_transport_causal_gate"]["points"])
        expected_cal = (cert["actors"][run]["calibration_gate_points"]
                        if run == "jvm/bash_agent-4o"
                        else dev["runs"][run]["calibration_transport_causal_gate"]["points"])
        if len(fit) != expected_fit or len(cal) != expected_cal:
            raise ValueError(f"Development point count drift: {run}")
        trained = [train(run, fit, seed) for seed in SEEDS]
        matrix = np.asarray([p["features"] for p in cal], dtype=np.float64)
        raw_residual = np.mean([model.predict(matrix, num_threads=1)
                                for model, _ in trained], axis=0)
        raw_base = np.asarray([p["transport_base_raw"] for p in cal], dtype=np.float64)
        actual = np.asarray([p["observed_input"] for p in cal], dtype=np.float64)
        prediction = np.maximum(0.0, raw_base + raw_residual)
        prediction_floor = np.floor(prediction)
        transport = raw_base + prior.FIT_MEDIAN_OFFSET[run]
        if not np.all(np.isfinite(prediction)):
            raise ValueError("Nonfinite residual-on-transport prediction")
        summary[run] = {
            "fit_points": len(fit), "calibration_points": len(cal),
            "fit_repositories": len({p["task_id"] for p in fit}),
            "calibration_repositories": len({p["task_id"] for p in cal}),
            "models": [meta for _, meta in trained],
            "residual_continuous": metrics(actual, prediction),
            "residual_floor": metrics(actual, prediction_floor),
            "transport_fit_offset": prior.FIT_MEDIAN_OFFSET[run],
            "transport_fit_constant": metrics(actual, transport),
        }
        for p, pred, pred_floor, residual in zip(cal, prediction, prediction_floor,
                                                 raw_residual, strict=True):
            all_rows.append({
                "actor": run, "task_id": p["task_id"], "call_index": p["call_index"],
                "observed_input": p["observed_input"],
                "transport_base_raw": p["transport_base_raw"],
                "transport_fit_median_input": p["transported_input"],
                "residual_prediction": float(residual),
                "residual_next_input_continuous": float(pred),
                "residual_next_input": int(pred_floor),
                "spent_before": p["spent_before"],
            })
    with POINTS.open("x", encoding="utf-8") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    equal_actor = {}
    for field in ("residual_continuous", "residual_floor", "transport_fit_constant"):
        equal_actor[field] = {key: float(np.mean([summary[run][field][key]
                                                for run in prior.ACTORS]))
                              for key in ("mae", "wape")}
    report = {
        "status": "four_actor_fit_cal_residual_on_transport_development_control",
        "chronology": "Additional strong same-information additive baseline trained before sealed test; calibration is development evidence",
        "test_trajectory_or_result_opened": False,
        "source_revision": lock["revision"],
        "identity_lock_sha256": prior.LOCK_SHA,
        "source_tree_metadata_sha256": prior.TREE_SHA,
        "development_qualification_sha256": prior.DEVELOPMENT_SHA,
        "budget_certificate_development_sha256": prior.CERTIFICATE_SHA,
        "script_sha256": digest(Path(__file__)),
        "tokenizer": {"package": "tiktoken", "version": tiktoken.__version__,
                      "encoding": "o200k_base"},
        "features": list(FEATURE_NAMES),
        "residual_feature_schema": "tool_tokens_transport_base_residual_v1",
        "feature_contract": "Original 21 causal prefix features, settled tool content tokens, raw additive transport base derived from these already-settled fields",
        "target": "observed next provider input minus transport_base_raw",
        "transport_base_raw": "previous provider input + previous provider output + tokenized received tool content; no fitted offset",
        "prediction": "max(0, transport_base_raw + mean(two seed residual predictions)); floor for discrete guard",
        "model": {"objective": "regression_l1", "rounds": 180,
                  "learning_rate": 0.05, "num_leaves": 15,
                  "min_data_in_leaf": 40, "feature_fraction": 0.9,
                  "bagging_fraction": 0.9, "bagging_freq": 1,
                  "lambda_l2": 1.0, "num_threads": 1,
                  "weight": "inverse eligible points per fit repository",
                  "seeds": list(SEEDS)},
        "calibration_points": {"path": POINTS.relative_to(ROOT).as_posix(),
                               "sha256": digest(POINTS), "rows": len(all_rows)},
        "runs": summary,
        "four_actor_equal_mean": equal_actor,
    }
    with REPORT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"report": REPORT.relative_to(ROOT).as_posix(),
                      "sha256": digest(REPORT),
                      "calibration_points_sha256": digest(POINTS),
                      "four_actor_equal_mean": equal_actor,
                      "per_actor": {run: {"fit": summary[run]["fit_points"],
                                          "cal": summary[run]["calibration_points"],
                                          "wape": summary[run]["residual_floor"]["wape"]}
                                    for run in prior.ACTORS}}, indent=2))


if __name__ == "__main__":
    main()
