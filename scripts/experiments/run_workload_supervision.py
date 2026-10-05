from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
import json
import importlib.metadata
from pathlib import Path
import shutil
import statistics
import sys
import time

import numpy as np

from token_prediction.estimators.neural_encoder import NeuralFeatureEncoder
from token_prediction.estimators.workload_mlp import (
    ARM_AUXILIARIES, AUXILIARY_NAMES, WorkloadFitConfig, WorkloadMLP, weighted_median,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "workspace/runs/workload-context-v1"
SEEDS = (20260719, 20260720, 20260721)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    from token_prediction.config import load_config
    from token_prediction.pipeline import (
        load_configured_experiment_context, load_experiment_trajectories,
    )
    from token_prediction.dataset.schema import PredictionPosition
    from token_prediction.dataset.workload import build_workload_labels, load_request_content_ledger
    from scripts.experiments.run_four_position_baseline_grid import capture_dirs

    cfg = load_config(args.config)
    roots = [args.runs_root, *getattr(args, "additional_runs_root", [])]
    captures = [capture for root in roots for capture in capture_dirs(root)]
    capture_by_trajectory = {}
    capture_sources = []
    for capture in captures:
        manifest = json.loads((capture / "capture.json").read_text(encoding="utf-8"))
        trajectory_id = f"dsh:{manifest['capture_id']}:{manifest['session_id']}"
        if trajectory_id in capture_by_trajectory:
            raise ValueError(f"duplicate capture identity: {trajectory_id}")
        capture_by_trajectory[trajectory_id] = capture
        capture_sources.append({
            "trajectory_id": trajectory_id, "capture_path": str(capture.resolve()),
            "sha256": {str(path.relative_to(capture)): sha256(path)
                       for path in sorted(capture.rglob("*"))
                       if path.is_file() and (path.name == "capture.json"
                                             or path.suffix in {".jsonl", ".zstd"})},
        })
    context = load_configured_experiment_context(cfg, captures)
    specs = [c.spec for c in context.experiment_cells
             if c.spec.position == PredictionPosition.TASK_UPDATE]
    if not specs:
        raise ValueError("configuration has no eligible Task Update cell")
    if len(specs) > 1 and not getattr(args, "merge_conditions", False):
        raise ValueError("multiple Task Update cells; explicitly request --merge-conditions for paired policies")
    spec = specs[0]
    recipe = next(c for c in spec.candidates if c.candidate_id == "task_update_ours")
    if any(next(c for c in item.candidates if c.candidate_id == "task_update_ours").feature_set
           != recipe.feature_set for item in specs):
        raise ValueError("all pooled conditions must use the same causal feature recipe")
    cohort = merge_workload_cohorts([
        context.dataset.select(item.position, item.target, required_features=item.required_features,
                               condition_id=item.condition_id,
                               token_unit_contract_id=item.token_unit_contract_id)
        for item in specs
    ])
    points_by_trajectory = {}
    for row in cohort.rows:
        points_by_trajectory.setdefault(row.point.trajectory_id, []).append(row.point)
    labels = {}
    for trajectory in load_experiment_trajectories(cfg.collection_source, captures):
        points = points_by_trajectory.get(trajectory.trajectory_id, [])
        if points:
            ledger = load_request_content_ledger(trajectory, capture_by_trajectory[trajectory.trajectory_id])
            for label in build_workload_labels(trajectory, points, request_ledger=ledger):
                labels[label.point_id] = label
    records = []
    for weighted in cohort.weighted_rows():
        row, point = weighted.row, weighted.row.point
        label = labels[point.point_id]
        horizon = {item.horizon: item for item in label.horizons}
        end = horizon[None].total_tokens.value
        if end != row.label:
            raise ValueError(f"workload End target differs from canonical target: {point.point_id}")
        values = {"attempts": label.future_attempts.value, "reuse": label.reuse.value}
        values.update({f"h{h}": horizon[h].total_tokens.value for h in (1, 2, 4, 8, 16)})
        features = dict(recipe.feature_set.select(point.features))
        for name in features:
            if name in AUXILIARY_NAMES or name.startswith("future_"):
                raise ValueError(f"future-label name in causal features: {name}")
        records.append({
            "point_id": point.point_id, "source_event_id": point.source_event_id,
            "task_id": point.task_id, "trajectory_id": point.trajectory_id,
            "run_id": point.run_id, "prediction_context_id": point.prediction_context_id,
            "condition_id": point.condition_id, "logical_call_id": point.logical_call_id,
            "attempt_id": point.attempt_id, "cutoff_event_seq": point.cutoff_event_seq,
            "position": point.position.value, "target": point.target.value,
            "known_offset_tokens": point.known_offset_tokens,
            "token_unit_contract_id": point.token_unit_contract_id,
            "features": features, "target_value": row.label,
            "weight": weighted.sample_weight, "auxiliary": values,
            "reuse_status": str(label.reuse.status), "reuse_reason": label.reuse.reason,
            "current_content_utf8_bytes": label.reuse.old_utf8_bytes,
            "future_reused_utf8_bytes": label.reuse.reused_utf8_bytes,
            "input_target": horizon[None].input_tokens.value,
            "output_target": horizon[None].output_tokens.value,
        })
    partitions = {}
    for seed in args.seeds:
        plan = context.split_plan_for(seed)
        partitions[str(seed)] = [
            {key: sorted(getattr(plan.partition(fold), key))
             for key in ("train_tasks", "validation_tasks", "calibration_tasks", "test_tasks")}
            for fold in range(plan.folds)
        ]
    data = {
        "schema_version": 1, "purpose": "development_only",
        "config_path": str(args.config.resolve()), "config_sha256": sha256(args.config),
        "runs_roots": [str(root.resolve()) for root in roots],
        "capture_sources": capture_sources, "feature_set_id": recipe.feature_set.feature_set_id,
        "condition_ids": sorted({r["condition_id"] for r in records}),
        "token_unit_contract_id": cohort.token_unit_contract_id,
        "n_tasks": len({r["task_id"] for r in records}), "n_points": len(records),
        "partitions": partitions, "records": records,
    }
    write_json(args.output / "dataset.json", data)
    write_json(args.output / "label-coverage.json", label_coverage(records))
    print(f"Prepared {data['n_tasks']} tasks / {len(records)} points", flush=True)
    return data


def merge_workload_cohorts(cohorts):

    if not cohorts:
        raise ValueError("at least one workload cohort is required")
    fields = ("position", "target", "token_unit_contract_id", "input_contract_id", "weighting_id")
    if any(any(getattr(item, key) != getattr(cohorts[0], key) for key in fields) for item in cohorts):
        raise ValueError("pooled workload cohorts must share position, target, unit, input and weighting contracts")
    rows = tuple(row for item in cohorts for row in item.rows)
    if len({row.point.point_id for row in rows}) != len(rows):
        raise ValueError("duplicate prediction point in pooled workload cohorts")
    return replace(cohorts[0], rows=rows, condition_id=cohorts[0].condition_id if len(cohorts) == 1
                   else "condition:pooled-workload-v1")


def label_coverage(records):

    coverage = {name: sum(r["auxiliary"][name] is not None for r in records)
                for name in AUXILIARY_NAMES}
    pairs = [(r["auxiliary"]["attempts"], r["auxiliary"]["reuse"])
             for r in records if all(r["auxiliary"][k] is not None for k in ("attempts", "reuse"))]
    unequal = sum(not np.isclose(n, m, rtol=1e-9, atol=1e-9) for n, m in pairs)
    reasons = {}
    for row in records:
        if row["auxiliary"]["reuse"] is None:
            reason = row["reuse_reason"]
            reasons[reason] = reasons.get(reason, 0) + 1
    return {"n_points": len(records), "observed": coverage, "reuse_missing_reasons": reasons,
            "paired_n_m_points": len(pairs), "n_m_differ_points": unequal,
            "reuse_scope": "client UTF-8 content-byte proxy; no provider token arithmetic",
            "interpretation": "M/N identifiability requires real selective-retention executions."}


def restore_point(record):
    from token_prediction.dataset.schema import PredictionPoint, PredictionPosition, PredictionTarget
    fields = {key: record[key] for key in (
        "point_id", "source_event_id", "task_id", "trajectory_id", "run_id",
        "prediction_context_id", "condition_id", "logical_call_id", "attempt_id",
        "cutoff_event_seq", "known_offset_tokens", "token_unit_contract_id")}
    return PredictionPoint(**fields, position=PredictionPosition(record["position"]),
                           target=PredictionTarget(record["target"]),
                           features={k: tuple(v) if isinstance(v, list) else v
                                     for k, v in record["features"].items()})


def validate_partitions(partition):
    groups = [set(partition[k]) for k in (
        "train_tasks", "validation_tasks", "calibration_tasks", "test_tasks")]
    if any(a & b for a, b in itertools.combinations(groups, 2)):
        raise ValueError("task leakage between split partitions")


def task_scores(predictions):
    per_task = {}
    for row in predictions:
        per_task.setdefault(row["task_id"], []).append(row)
    result = {}
    for task, rows in per_task.items():
        ws = np.asarray([r["weight"] for r in rows])
        y = np.asarray([r["target"] for r in rows])
        pred = np.asarray([r["prediction"] for r in rows])
        result[task] = (float(np.average(abs(y - pred), weights=ws)),
                        float(np.average(y, weights=ws)))
    return result


def aggregate(predictions):
    scores = task_scores(predictions)
    mae = statistics.fmean(s[0] for s in scores.values())
    denominator = statistics.fmean(s[1] for s in scores.values())
    return {"mae": mae, "wape": mae / denominator if denominator else None,
            "n_tasks": len(scores), "n_predictions": len(predictions)}


def comparison(a, b, seed=20260915):
    def aligned(rows):
        result = {}
        for row in rows:
            key = (row["seed"], row["fold"], row["point_id"])
            if key in result:
                raise ValueError("duplicate point in paired comparison")
            result[key] = (row["task_id"], row["target"], row["weight"])
        return result

    if aligned(a) != aligned(b):
        raise ValueError("paired comparison requires identical points, targets, and weights")
    aa, bb = task_scores(a), task_scores(b)
    if set(aa) != set(bb):
        raise ValueError("paired comparison requires identical task sets")
    keys = sorted(aa)
    av = np.asarray([aa[k][0] for k in keys])
    bv = np.asarray([bb[k][0] for k in keys])
    draws = np.random.default_rng(seed).integers(0, len(keys), size=(10000, len(keys)))
    delta = (av - bv)[draws].mean(axis=1)
    return {"delta_mae": float((av - bv).mean()),
            "delta_mae_ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
            "relative_reduction": float(1 - av.mean() / bv.mean()) if bv.mean() else None,
            "n_tasks": len(keys)}


def tuning_configs(arm, max_epochs, seed):
    for lr in (0.0003, 0.001, 0.003):
        for variable in ((0, 0.0001, 0.01) if arm == "direct" else (0.1, 1, 10)):
            yield WorkloadFitConfig(learning_rate=lr, max_epochs=max_epochs, seed=seed,
                                    weight_decay=variable if arm == "direct" else 0.0001,
                                    auxiliary_weight=1.0 if arm == "direct" else variable)


def common_cohort_weights(records, eligible):

    from token_prediction.dataset.schema import DatasetRow, DatasetSlice, LabelStatus

    selected = [r for r, keep in zip(records, eligible) if keep]
    if not selected:
        raise ValueError("empty common cohort")
    first = restore_point(selected[0])
    cohort = DatasetSlice(
        dataset_id="workload-common-cohort-v1", position=first.position, target=first.target,
        condition_id=first.condition_id, eligibility_policy_id="observed-workload-labels-v1",
        rows=tuple(DatasetRow(restore_point(r), int(r["target_value"]), LabelStatus.OBSERVED)
                   for r in selected), token_unit_contract_id=first.token_unit_contract_id,
    )
    mapping = {wr.row.point.point_id: wr.sample_weight for wr in cohort.weighted_rows()}
    return np.asarray([mapping.get(r["point_id"], 0.0) for r in records])


def execute(args, data):
    from lightgbm import LGBMRegressor

    records = data["records"]
    if data.get("purpose") != "development_only":
        raise ValueError("this runner is restricted to development data")
    if len({r["token_unit_contract_id"] for r in records}) != 1:
        raise ValueError("provider unit contracts must be evaluated separately")
    points = [restore_point(r) for r in records]
    tasks = np.asarray([r["task_id"] for r in records])
    y = np.asarray([r["target_value"] for r in records], dtype=float)
    weights = np.asarray([r["weight"] for r in records], dtype=float)
    aux = np.asarray([[r["auxiliary"][name] for name in AUXILIARY_NAMES]
                      for r in records], dtype=float)
    available, skipped = [], {}
    for arm, names in ARM_AUXILIARIES.items():
        absent = [name for name in names if not np.isfinite(aux[:, AUXILIARY_NAMES.index(name)]).any()]
        if absent:
            skipped[arm] = {"status": "not_evaluated", "reason": "missing auxiliary labels",
                            "missing": absent}
        else:
            available.append(arm)

    eligible = np.ones(len(records), dtype=bool)
    for arm in available:
        for name in ARM_AUXILIARIES[arm]:
            eligible &= np.isfinite(aux[:, AUXILIARY_NAMES.index(name)])
    if not eligible.any():
        raise ValueError("no common cohort with observed required labels")
    weights = common_cohort_weights(records, eligible)
    reports, predictions = [], []
    for seed in args.seeds:
        for fold, part in enumerate(data["partitions"][str(seed)]):
            validate_partitions(part)
            indices = {key: np.flatnonzero(eligible & np.isin(tasks, part[f"{key}_tasks"]))
                       for key in ("train", "validation", "test")}
            tr, va, te = (indices[k] for k in ("train", "validation", "test"))
            if min(len(tr), len(va), len(te)) == 0:
                raise ValueError(f"empty common-cohort partition at seed {seed} fold {fold}")
            enc = NeuralFeatureEncoder.fit([points[i] for i in tr])
            x = enc.transform(points).matrix.astype(np.float32)
            package = args.output / "models" / f"seed-{seed}-fold-{fold}"
            package.mkdir(parents=True, exist_ok=True)
            write_json(package / "encoder.json", enc.to_dict())
            for arm in (*available, "lightgbm", "history_median"):
                start = time.perf_counter()
                fit = {"seed": seed, "fold": fold, "arm": arm}
                if arm in available:
                    candidates = []
                    best_model = None
                    for config in tuning_configs(arm, args.max_epochs, seed + fold):
                        model = WorkloadMLP(x.shape[1], config).fit(
                            x[tr], y[tr], weights[tr], aux[tr], arm=arm,
                            validation_x=x[va], validation_y=y[va],
                            validation_weights=weights[va])
                        candidates.append({"config": asdict(config), **model.fit_report})
                        if best_model is None or model.fit_report["validation_mae"] < best_model.fit_report["validation_mae"]:
                            best_model = model
                    best_model.save(package / f"{arm}.pt")
                    estimate = best_model.predict(x[te])
                    fit.update({"candidates": candidates, "selected": best_model.fit_report,
                                "config": asdict(best_model.config)})
                elif arm == "lightgbm":
                    best_model, best_score, candidates = None, float("inf"), []
                    for leaves, minimum in itertools.product((7, 15, 31), (5, 15, 30)):
                        model = LGBMRegressor(objective="regression_l1", num_leaves=leaves,
                                              min_child_samples=minimum, n_estimators=200,
                                              learning_rate=0.05, random_state=seed + fold,
                                              verbosity=-1, n_jobs=1, deterministic=True,
                                              force_col_wise=True)
                        model.fit(x[tr], y[tr], sample_weight=weights[tr])
                        vp = np.maximum(0, model.predict(x[va]))
                        score = float(np.average(abs(vp - y[va]), weights=weights[va]))
                        candidates.append({"num_leaves": leaves, "min_child_samples": minimum,
                                           "validation_mae": score})
                        if score < best_score:
                            best_model, best_score = model, score
                    estimate = np.maximum(0, best_model.predict(x[te]))
                    best_model.booster_.save_model(str(package / "lightgbm.txt"))
                    fit.update({"candidates": candidates, "validation_mae": best_score})
                else:
                    estimate = np.full(len(te), weighted_median(y[tr], weights[tr]))
                fit["fit_seconds"] = time.perf_counter() - start
                reports.append(fit)
                for i, estimate_i in zip(te, estimate):
                    predictions.append({"seed": seed, "fold": fold, "arm": arm,
                                        "point_id": records[i]["point_id"], "task_id": tasks[i],
                                        "trajectory_id": records[i]["trajectory_id"],
                                        "target": float(y[i]), "prediction": float(estimate_i),
                                        "weight": float(weights[i])})
                print(f"seed {seed} fold {fold} {arm}: complete ({fit['fit_seconds']:.1f}s)", flush=True)
            write_json(args.output / "fit-reports.json", reports)
            with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
                for row in predictions:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    by_arm = {arm: [r for r in predictions if r["arm"] == arm]
              for arm in (*available, "lightgbm", "history_median")}
    metrics = {arm: aggregate(rows) for arm, rows in by_arm.items()}
    comparisons = {f"{arm}_vs_direct": comparison(rows, by_arm["direct"])
                   for arm, rows in by_arm.items() if arm != "direct"}
    if "direct_n" in by_arm and "direct_horizons" in by_arm:
        comparisons["direct_n_vs_direct_horizons"] = comparison(by_arm["direct_n"], by_arm["direct_horizons"])
    if "direct_nm" in by_arm:
        comparisons["direct_nm_vs_direct_n"] = comparison(by_arm["direct_nm"], by_arm["direct_n"])
        comparisons["direct_nm_vs_direct_horizons"] = comparison(by_arm["direct_nm"], by_arm["direct_horizons"])
    limitations = ["Development cohort; final holdout remains closed.",
                   "Context-policy transfer and online budget control are unevaluated."]
    if "direct_nm" in skipped:
        limitations.append("Missing reuse labels cannot establish the joint-supervision hypothesis.")
    result = {"status": "development_diagnostic", "dataset_sha256": sha256(args.output / "dataset.json"),
              "n_tasks": len(set(tasks[eligible])), "n_points": int(eligible.sum()),
              "split_seeds": args.seeds, "metrics": metrics, "comparisons": comparisons,
              "unevaluated": skipped, "fit_reports": "fit-reports.json",
              "prediction_file": "predictions.jsonl", "token_unit_contract_id": data["token_unit_contract_id"],
              "condition_ids": sorted({r["condition_id"] for i, r in enumerate(records) if eligible[i]}),
              "limitations": limitations}
    write_json(args.output / "results.json", result)
    lines = ["# TokenCast workload supervision: development diagnostic", "",
             f"{result['n_tasks']} tasks, {result['n_points']} Task Update points; seeds {args.seeds}.",
             "Provider-native remaining total; task-balanced MAE and WAPE.", "",
             "| Arm | MAE | WAPE |", "|---|---:|---:|"]
    lines += [f"| {arm} | {m['mae']:,.1f} | {m['wape']:.4f} |" for arm, m in metrics.items()]
    lines += ["", "## Paired task bootstrap", "",
              "Negative MAE differences favor the named arm; intervals are pointwise 95% diagnostics.", ""]
    lines += [f"- {name}: {v['delta_mae']:,.1f}, CI {v['delta_mae_ci95']}" for name, v in comparisons.items()]
    lines += ["", "## Unevaluated", ""] + [f"- {arm}: missing {entry['missing']} labels." for arm, entry in skipped.items()]
    lines += ["", *result["limitations"], "", "Artifacts: dataset.json, fit-reports.json, predictions.jsonl, models/."]
    (args.output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/workload_context_v1.toml")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "workspace/datasets/swebench-dual-model-v1/recollection-gpt-oauth-v18/runs/runs")
    parser.add_argument("--additional-runs-root", type=Path, action="append", default=[])
    parser.add_argument("--merge-conditions", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-prepared", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.json").exists():
        raise ValueError("completed output directory already exists; choose a new directory")
    data = json.loads((args.output / "dataset.json").read_text(encoding="utf-8")) if args.reuse_prepared else prepare(args)
    if not args.prepare_only:
        snapshot = args.output / "source"
        snapshot.mkdir(exist_ok=True)
        code = {}
        for relative in (
            "scripts/experiments/run_workload_supervision.py",
            "src/token_prediction/estimators/workload_mlp.py",
            "src/token_prediction/estimators/neural_encoder.py",
            "src/token_prediction/dataset/workload.py",
            "src/token_prediction/dataset/schema.py",
        ):
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
            code[relative] = sha256(destination)
        write_json(args.output / "run.json", {
            "schema_version": 1, "status": "running", "purpose": "development_only",
            "dataset_sha256": sha256(args.output / "dataset.json"), "code_sha256": code,
            "python": sys.version, "packages": {name: importlib.metadata.version(name)
                for name in ("numpy", "torch", "lightgbm", "scikit-learn")},
            "arguments": json.loads(json.dumps(vars(args), default=str)),
        })
        execute(args, data)
        manifest = json.loads((args.output / "run.json").read_text(encoding="utf-8"))
        manifest["status"] = "completed"
        manifest["artifact_sha256"] = {
            str(path.relative_to(args.output)): sha256(path)
            for path in args.output.rglob("*")
            if path.is_file() and path.name != "run.json"
        }
        write_json(args.output / "run.json", manifest)


if __name__ == "__main__":
    main()
