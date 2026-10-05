from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .develop_envbench_jvm_budget_models import (
    BASE, CAPS_FILE, CAPS_SHA, LOCK, LOCK_SHA, ROOT, V2, V2_SHA,
    development_data, digest,
)
from .develop_envbench_b50_overcap_classifier import aggregate, replay


SOURCE = ROOT / "workspace/experiments/envbench_b50_overcap_dev_20260923/report.json"
SOURCE_SHA = "1531a771b16f15d34ee5339e6001d8a1a12c8df6ce90587601bc44a1fccf731e"
QMAX = ROOT / "workspace/experiments/envbench_b50_overcap_qmax_dev_20260923/report.json"
QMAX_SHA = "4a1e2fbeb34b55101ab84269badd3a078e64a77a2314bb28ec4641970341a2af"
OUT = ROOT / "workspace/experiments/envbench_b50_probability_guards_dev_20260923/report.json"
CLASSIFIERS = ("overcap_classifier_all", "overcap_classifier_complete_only")
SEEDS = (20260923, 20270923)
GUARDS = (0.95, 0.97)


def main() -> None:
    for path, expected in ((LOCK, LOCK_SHA), (V2, V2_SHA),
                           (CAPS_FILE, CAPS_SHA), (SOURCE, SOURCE_SHA), (QMAX, QMAX_SHA)):
        if digest(path) != expected:
            raise ValueError(f"Frozen development source changed: {path}")
    if OUT.exists():
        raise FileExistsError(OUT)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    qmax = json.loads(QMAX.read_text(encoding="utf-8"))
    cap = json.loads(CAPS_FILE.read_text(encoding="utf-8"))["caps"]["0.5"]
    if cap != 47078:
        raise ValueError("B50 drift")
    points, tasks = development_data(lock, v2)
    cal = [point for point in points["calibration"] if point["spent"] < cap]
    matrix = np.asarray([point["features"] for point in cal], dtype=float)
    if matrix.shape != (1180, 21):
        raise ValueError("Causal feature matrix drift")
    for arm in CLASSIFIERS:
        models = []
        for seed in SEEDS:
            item = source["models"][arm][str(seed)]
            path = ROOT / item["path"]
            if digest(path) != item["sha256"]:
                raise ValueError("Classifier artifact drift")
            models.append(lgb.Booster(model_file=str(path)))
        prediction = np.mean([model.predict(matrix, num_threads=1)
                              for model in models], axis=0)
        for point, value in zip(cal, prediction, strict=True):
            point.setdefault("score", {})[arm] = float(value)

        original = aggregate(replay(cal, tasks, arm, cap,
                                    qmax["settings"][arm]["q_final"]))["all"]
        if original != qmax["summaries"][arm]["all"]:
            raise ValueError("qmax classifier replay mismatch")
    results = {}
    comparisons = {}
    direct = qmax["per_task_rows"]["direct_remainder_two_seed"]
    reference = np.asarray([row["reference_tokens"] for row in direct], dtype=float)
    direct_saved = np.asarray([row["saved_tokens"] for row in direct], dtype=float)
    rng = np.random.default_rng(20260923)
    sample = rng.integers(0, len(reference), size=(10000, len(reference)))
    denominator = reference[sample].sum(axis=1)
    for arm in CLASSIFIERS:
        results[arm] = {}
        comparisons[arm] = {}
        for floor in GUARDS:
            q = max(qmax["settings"][arm]["q_final"], floor)
            rows = replay(cal, tasks, arm, cap, q)
            summary = aggregate(rows)
            if summary["complete_undercap"]["false_aborts"] != 0:
                raise ValueError("Guard produced calibration false abort")
            key = str(floor)
            results[arm][key] = {"q": q, "summary": summary, "per_task_rows": rows}
            saved = np.asarray([row["saved_tokens"] for row in rows], dtype=float)
            delta = saved - direct_saved
            draws = delta[sample].sum(axis=1) / denominator
            comparisons[arm][key] = {
                "versus_direct_qmax_pooled_delta": float(sum(delta) / sum(reference)),
                "bootstrap_two_sided_95": [float(np.quantile(draws, .025)),
                                            float(np.quantile(draws, .975))],
                "bootstrap_one_sided_lower_95": float(np.quantile(draws, .05)),
            }
    OUT.parent.mkdir(parents=True)
    report = {
        "status": "post_hoc_calibration_only_probability_guard_sensitivity",
        "chronology": "0.95 and 0.97 proposed after observing 95%-rank and qmax calibration outcomes",
        "source_report_sha256": SOURCE_SHA, "qmax_report_sha256": QMAX_SHA,
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "test_trajectory_access": False, "run_level_results_access": False,
        "cap": cap,
        "guard_rule": "For binary probability arms q=max(original qmax, fixed probability floor); regression controls retain own qmax because scores have different units",
        "direct_remainder_qmax_summary": qmax["summaries"]["direct_remainder_two_seed"]["all"],
        "previous_input_qmax_summary": qmax["summaries"]["previous_input"]["all"],
        "results": results, "paired_comparisons": comparisons,
    }
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report_sha256": digest(OUT),
                      "B50": {arm: {floor: value["summary"]["all"]
                                     for floor, value in results[arm].items()}
                              for arm in CLASSIFIERS},
                      "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
