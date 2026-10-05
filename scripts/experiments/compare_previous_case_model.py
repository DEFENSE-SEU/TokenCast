import argparse
import json
from pathlib import Path
import numpy as np
import lightgbm as lgb
from token_prediction.case_study.evidence import load_run_evidence


def matrix(d):
    known = np.asarray(d["input_known"], dtype=bool)
    return np.column_stack(
        (
            d["task"],
            d["state"],
            d["observation"],
            np.where(known, d["input_length"], 0),
            known.astype(float),
            d["spent"],
            d["allowance"],
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", type=Path, default=Path("workspace/runs/composable-case-pilot"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("runs", nargs="+", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    with np.load(a.pilot / "segments.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    provenance = json.loads((a.pilot / "provenance.json").read_text(encoding="utf-8"))
    partition = provenance["task_family_partition"]
    membership = np.asarray([partition[str(f)] for f in d["task_family_id"]])
    X = matrix(d)
    params = dict(
        objective="quantile",
        alpha=0.5,
        learning_rate=0.03,
        num_leaves=15,
        min_data_in_leaf=10,
        max_depth=6,
        lambda_l2=1.0,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
        seed=20260919,
        num_threads=1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    models = {}
    summaries = {}
    calibration = []
    for point in ["task_start", "call_start", "in_call_update", "task_update"]:
        call = point in ["call_start", "in_call_update"]
        selected = (d["position"] == point) & (d["n"] > 0)
        selected &= (d["horizon"] == 1) if call else (d["horizon"] == d["allowance"])
        seen = set()
        indices = []
        for i in np.flatnonzero(selected):
            key = str(d["point_id"][i])
            if key not in seen:
                indices.append(i)
                seen.add(key)
        indices = np.asarray(indices, dtype=int)
        y = d["label_call_cost" if call else "label_cost"]
        ids = {
            name: indices[membership[indices] == name]
            for name in ["train", "validation", "calibration"]
        }
        datasets = {}
        for name in ["train", "validation"]:
            ix = ids[name]
            families, counts = np.unique(d["task_family_id"][ix], return_counts=True)
            counts = dict(zip(families, counts))
            weights = np.asarray([1 / counts[f] for f in d["task_family_id"][ix]])
            weights /= weights.mean()
            datasets[name] = lgb.Dataset(X[ix], label=y[ix], weight=weights)
        model = lgb.train(
            params,
            datasets["train"],
            num_boost_round=300,
            valid_sets=[datasets["validation"]],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        models[point] = model
        model.save_model(str(a.output / f"{point}.txt"))
        rows = [
            dict(
                prediction_point=point,
                point_id=str(d["point_id"][i]),
                task_family_id=str(d["task_family_id"][i]),
                actual=float(y[i]),
                prediction=max(0.0, float(pred)),
            )
            for i, pred in zip(ids["calibration"], model.predict(X[ids["calibration"]]))
        ]
        calibration.extend(rows)
        families = sorted(set(r["task_family_id"] for r in rows))
        mae = float(np.mean([abs(r["prediction"] - r["actual"]) for r in rows]))
        macro = float(
            np.mean(
                [
                    np.mean(
                        [
                            abs(r["prediction"] - r["actual"])
                            for r in rows
                            if r["task_family_id"] == f
                        ]
                    )
                    for f in families
                ]
            )
        )
        summaries[point] = dict(
            n_points=len(rows),
            n_tasks=len(families),
            point_mae=mae,
            task_macro_mae=macro,
            best_iteration=model.best_iteration,
            train_points=len(ids["train"]),
            validation_points=len(ids["validation"]),
        )
    cases = []
    for run in a.runs:
        for r in load_run_evidence(
            run, allow_ineligible_terminal=True,
            request_surface_features=provenance.get("feature_schema") == "causal_request_surface_v2",
        ):
            point = r["prediction_point"]
            pred = max(0.0, float(models[point].predict(matrix(r["core_query"]))[0]))
            actual = r["labels"]["actual_tokens"]
            cases.append(
                dict(
                    run=str(run),
                    point_id=r["point_id"],
                    prediction_point=point,
                    actual=actual,
                    prediction=pred,
                    absolute_error=abs(pred - actual),
                    scope=r["labels"]["actual_scope"],
                    run_metadata=r["run_metadata"],
                )
            )
    report = dict(
        control="new independent LightGBM heads; same causal inputs and latest full-cost targets",
        original_checkpoint_status="unsupported new actor condition/token unit; not deployed",
        task_family_partition=partition,
        parameters=params,
        calibration_summary=summaries,
        calibration_predictions=calibration,
        case_predictions=cases,
        limitations=[
            "Calibration diagnostics, no untouched-test claim.",
            "Source actor differs from case actors.",
            "Budget-stopped cases measure capped execution only.",
            "No case-based hyperparameter selection.",
        ],
    )
    (a.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(calibration_summary=summaries, case_predictions=cases), indent=2))


if __name__ == "__main__":
    main()
