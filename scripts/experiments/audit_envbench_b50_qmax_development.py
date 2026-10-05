from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .develop_envbench_jvm_budget_models import (
    BASE, CAPS_FILE, CAPS_SHA, LOCK, LOCK_SHA, ROOT, V2, V2_SHA,
    development_data, digest,
)
from .develop_envbench_b50_overcap_classifier import (
    ARM_NAMES, DIRECT_REPORT, DIRECT_REPORT_SHA, FIRST_REPORT, FIRST_REPORT_SHA,
    OUT as ORIGINAL_DIR, aggregate, load_existing_booster, replay, threshold,
)


ORIGINAL_REPORT = ORIGINAL_DIR / "report.json"
ORIGINAL_REPORT_SHA = "1531a771b16f15d34ee5339e6001d8a1a12c8df6ce90587601bc44a1fccf731e"
NEW_DIR = ROOT / "workspace/experiments/envbench_b50_overcap_qmax_dev_20260923"
SEEDS = (20260923, 20270923)


def model_from_receipt(item: dict, *, old_windows_text: bool) -> lgb.Booster:
    path = ROOT / item["path"]
    if digest(path) != item["sha256"]:
        raise ValueError(f"Model source hash drift: {path}")
    if old_windows_text:
        return load_existing_booster(path)
    return lgb.Booster(model_file=str(path))


def main() -> None:
    for path, expected in ((LOCK, LOCK_SHA), (V2, V2_SHA),
                           (CAPS_FILE, CAPS_SHA), (FIRST_REPORT, FIRST_REPORT_SHA),
                           (DIRECT_REPORT, DIRECT_REPORT_SHA),
                           (ORIGINAL_REPORT, ORIGINAL_REPORT_SHA)):
        if digest(path) != expected:
            raise ValueError(f"Frozen development source changed: {path}")
    if NEW_DIR.exists():
        raise FileExistsError(NEW_DIR)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    original = json.loads(ORIGINAL_REPORT.read_text(encoding="utf-8"))
    first = json.loads(FIRST_REPORT.read_text(encoding="utf-8"))
    direct = json.loads(DIRECT_REPORT.read_text(encoding="utf-8"))
    cap = json.loads(CAPS_FILE.read_text(encoding="utf-8"))["caps"]["0.5"]
    if cap != 47078:
        raise ValueError("B50 drift")
    points, tasks = development_data(lock, v2)
    cal = [point for point in points["calibration"] if point["spent"] < cap]
    matrix = np.asarray([point["features"] for point in cal], dtype=float)
    if matrix.shape != (1180, 21):
        raise ValueError("Calibration feature matrix drift")
    for arm in ("overcap_classifier_all", "overcap_classifier_complete_only"):
        models = [model_from_receipt(original["models"][arm][str(seed)],
                                     old_windows_text=False) for seed in SEEDS]
        prediction = np.mean([model.predict(matrix, num_threads=1) for model in models], axis=0)
        for point, value in zip(cal, prediction, strict=True):
            point.setdefault("score", {})[arm] = float(value)
    direct_models = [model_from_receipt(direct["models"][str(seed)],
                                        old_windows_text=True) for seed in SEEDS]
    direct_prediction = np.mean([
        np.maximum(0.0, np.expm1(model.predict(matrix, num_threads=1)))
        for model in direct_models], axis=0)
    next_model = model_from_receipt(first["models"]["next_input"], old_windows_text=True)
    next_prediction = np.maximum(0.0, np.expm1(next_model.predict(matrix, num_threads=1)))
    for row, point in enumerate(cal):
        for arm, value in (("direct_remainder_two_seed", direct_prediction[row]),
                           ("direct_next_input", next_prediction[row]),
                           ("previous_input", point["previous_input"])):
            point["score"][arm] = float((point["spent"] + value - cap) / cap)

    for arm in ARM_NAMES:
        original_q = threshold(cal, tasks, arm, cap)
        if abs(original_q["q_final"] - original["thresholds"][arm]["q_final"]) > 1e-10:
            raise ValueError("95%-rank threshold replication failed")
        old_summary = aggregate(replay(cal, tasks, arm, cap, original_q["q_final"]))["all"]
        if old_summary != original["summaries"][arm]["all"]:
            raise ValueError("95%-rank replay replication failed")
    undercap_ids = sorted(task_id for task_id, task in tasks.items()
                          if task["group"] == "calibration" and
                          task["terminal_class"] == "model_stop" and
                          task["nominal_total_tokens"] <= cap)
    if len(undercap_ids) != 64:
        raise ValueError("Undercap risk set drift")
    by_id = defaultdict(list)
    for point in cal:
        by_id[point["task_id"]].append(point)
    settings, rows, summaries = {}, {}, {}
    for arm in ARM_NAMES:
        maxima = [max((point["score"][arm] for point in by_id[task_id]),
                      default=-float("inf")) for task_id in undercap_ids]
        if not all(np.isfinite(value) for value in maxima):
            raise ValueError("No-call undercap case makes qmax undefined")
        q_raw = max(maxima)
        q = max(0.0, q_raw)
        settings[arm] = {"q_raw": q_raw, "q_final": q,
                         "risk_calibration_cases": len(undercap_ids),
                         "argmax_task_id": undercap_ids[int(np.argmax(maxima))]}
        rows[arm] = replay(cal, tasks, arm, cap, q)
        summaries[arm] = aggregate(rows[arm])
        if summaries[arm]["complete_undercap"]["false_aborts"] != 0:
            raise ValueError("qmax failed zero calibration false abort")
    reference = np.asarray([row["reference_tokens"] for row in rows[ARM_NAMES[0]]], dtype=float)
    if np.any(reference <= 0):
        raise ValueError("Nonpositive reference cost")
    rng = np.random.default_rng(20260923)
    sample = rng.integers(0, len(reference), size=(10000, len(reference)))
    denominator = reference[sample].sum(axis=1)
    comparison = {}
    for a, b in (("overcap_classifier_all", "direct_remainder_two_seed"),
                 ("overcap_classifier_complete_only", "direct_remainder_two_seed"),
                 ("overcap_classifier_all", "overcap_classifier_complete_only"),
                 ("overcap_classifier_all", "previous_input")):
        delta = np.asarray([left["saved_tokens"] - right["saved_tokens"]
                            for left, right in zip(rows[a], rows[b], strict=True)], dtype=float)
        draws = delta[sample].sum(axis=1) / denominator
        comparison[f"{a}_minus_{b}"] = {
            "paired_pooled_delta": float(sum(delta) / sum(reference)),
            "bootstrap_two_sided_95": [float(np.quantile(draws, .025)),
                                        float(np.quantile(draws, .975))],
            "bootstrap_one_sided_lower_95": float(np.quantile(draws, .05)),
        }
    NEW_DIR.mkdir(parents=True)
    output = {
        "status": "B50_qmax_calibration_only_post_hoc_sensitivity",
        "chronology": "After observing original 95%-rank calibration false aborts; no test or results access",
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "original_report_sha256": ORIGINAL_REPORT_SHA,
        "source_hashes": {str(path.relative_to(ROOT)): expected for path, expected in (
            (LOCK, LOCK_SHA), (V2, V2_SHA), (CAPS_FILE, CAPS_SHA),
            (FIRST_REPORT, FIRST_REPORT_SHA), (DIRECT_REPORT, DIRECT_REPORT_SHA))},
        "test_trajectory_access": False, "run_level_results_access": False,
        "cap": cap,
        "rule": "q=max(0,max over 64 complete T<=B50 case-max scores); strict score>q; same for all arms",
        "settings": settings, "summaries": summaries,
        "comparisons": comparison,
        "per_task_rows": {arm: rows[arm] for arm in ARM_NAMES},
    }
    path = NEW_DIR / "report.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report_sha256": digest(path),
                      "B50": {arm: summaries[arm]["all"] for arm in ARM_NAMES},
                      "comparisons": comparison}, indent=2))


if __name__ == "__main__":
    main()
