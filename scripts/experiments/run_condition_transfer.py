from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from scripts.experiments.run_four_position_baseline_grid import SPLIT_SEEDS, capture_dirs
from token_prediction.config import load_config
from token_prediction.estimators import FitContext, builtin_registry
from token_prediction.evaluation.calibration import CalibrationExample
from token_prediction.evaluation.calibrator_registry import fit_calibrator
from token_prediction.experiments.candidates import _predict_rows, make_training_view
from token_prediction.pipeline import load_configured_experiment_context

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "experiments" / "four_position_main_table_v1.toml"
SHARED_CONDITION = "transfer"
ITERATIONS = 10_000
DRAWS = 3


def relabel(rows):
    return [replace(row, point=replace(row.point, condition_id=SHARED_CONDITION)) for row in rows]


def slice_for(context, cell):
    spec = cell.spec
    return context.dataset.select(
        spec.position, spec.target,
        required_features=spec.required_features,
        condition_id=spec.condition_id,
        token_unit_contract_id=spec.token_unit_contract_id,
    )


def fit_predict(registry, candidate, alpha, calibrator_id, *, slice_, train, validation, calibration, test, weights, seed, fold):
    estimator = registry.create(candidate.estimator_id, candidate.params)
    fitted = estimator.fit(
        make_training_view(slice_, train, weights, candidate.feature_set),
        make_training_view(slice_, validation, weights, candidate.feature_set),
        FitContext(seed=seed, fold=fold, interval_alpha=alpha, checkpoint=None),
    )
    cal_forecasts = _predict_rows(fitted, calibration, dataset_slice=slice_, feature_set=candidate.feature_set)
    calibrator = fit_calibrator(
        calibrator_id, alpha=alpha,
        examples=[CalibrationExample(task_id=r.point.task_id, forecast=cal_forecasts[r.point.point_id], target_value=float(r.label), sample_weight=weights[r.point.point_id]) for r in calibration if r.label is not None],
    )
    forecasts = _predict_rows(fitted, test, dataset_slice=slice_, feature_set=candidate.feature_set)
    return {pid: calibrator.transform(f) for pid, f in forecasts.items()}


def score(forecasts, rows, weights):
    per_task_err: dict[str, list[tuple[float, float]]] = defaultdict(list)
    per_task_cov: dict[str, list[tuple[float, float]]] = defaultdict(list)
    per_task_width: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        f = forecasts[row.point.point_id]
        y = float(row.label)
        w = weights[row.point.point_id]
        per_task_err[row.point.task_id].append((w, abs(f.point - y)))
        per_task_cov[row.point.task_id].append((w, float(f.lower <= y <= f.upper)))
        per_task_width[row.point.task_id].append((w, (f.upper - f.lower) / max(y, 1.0)))
    wmean = lambda items: sum(w * v for w, v in items) / sum(w for w, _ in items)
    task_mae = {t: wmean(v) for t, v in per_task_err.items()}
    truth = sum(w * float(r.label) for r in rows for w in [weights[r.point.point_id]])
    return {
        "task_mae": task_mae,
        "mae": statistics.fmean(task_mae.values()),
        "wape": sum(w * e for items in per_task_err.values() for w, e in items) / truth,
        "coverage": statistics.fmean(wmean(v) for v in per_task_cov.values()),
        "normalized_width": statistics.fmean(wmean(v) for v in per_task_width.values()),
    }


def paired(a: dict[str, float], b: dict[str, float]) -> dict:
    tasks = sorted(set(a) & set(b))
    diff = np.asarray([a[t] - b[t] for t in tasks])
    draws = np.random.default_rng(0).integers(0, len(tasks), size=(ITERATIONS, len(tasks)))
    means = diff[draws].mean(axis=1)
    return {"n_tasks": len(tasks), "delta": float(diff.mean()), "relative": float(diff.mean() / np.mean([b[t] for t in tasks])), "ci_lower": float(np.percentile(means, 2.5)), "ci_upper": float(np.percentile(means, 97.5)), "win_probability": float((means < 0).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--shots", default="5,10,20")
    parser.add_argument("--candidates", default="", help="comma list; default = every non-lifecycle candidate")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    source = load_configured_experiment_context(config, capture_dirs(args.source))
    target = load_configured_experiment_context(config, capture_dirs(args.target))
    registry = builtin_registry()
    shots = [int(s) for s in args.shots.split(",") if s]
    wanted = set(args.candidates.split(",")) if args.candidates else None

    source_cells = {cell.spec.experiment_id.split("--")[0]: cell for cell in source.experiment_cells if cell.spec.token_unit_contract_id}
    results: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))


    partial_dir = args.output / "partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    finished_blocks: set[tuple[str, int]] = set()
    for path in sorted(partial_dir.glob("*.json")):
        block = json.loads(path.read_text(encoding="utf-8"))
        for cid, by_regime in block["scores"].items():
            for name, s in by_regime.items():
                results[block["prefix"]][cid][name][block["seed"]] = s
        finished_blocks.add((block["prefix"], block["seed"]))
        print(f"resumed {block['prefix']} seed {block['seed']} from {path.name}", flush=True)
    for cell in target.experiment_cells:
        spec = cell.spec
        prefix = spec.experiment_id.split("--")[0]
        if spec.token_unit_contract_id is None or prefix not in source_cells:
            continue
        t_slice, s_slice = slice_for(target, cell), slice_for(source, source_cells[prefix])
        t_rows, s_rows = relabel([r for r in t_slice.rows if r.eligible]), relabel([r for r in s_slice.rows if r.eligible])
        weights = {i.row.point.point_id: i.sample_weight for i in t_slice.weighted_rows()}
        weights.update({i.row.point.point_id: i.sample_weight for i in s_slice.weighted_rows()})

        t_slice = replace(t_slice, condition_id=SHARED_CONDITION, rows=tuple(t_rows))
        by_task_t: dict[str, list] = defaultdict(list)
        by_task_s: dict[str, list] = defaultdict(list)
        for r in t_rows:
            by_task_t[r.point.task_id].append(r)
        for r in s_rows:
            by_task_s[r.point.task_id].append(r)
        target_tasks = set(by_task_t)
        candidates = [c for c in spec.candidates if not c.graph.is_lifecycle and (wanted is None or c.candidate_id in wanted)]
        for seed in SPLIT_SEEDS:
            if (prefix, seed) in finished_blocks:
                continue
            plan = target.split_plan_for(seed)
            regimes: dict[str, dict] = defaultdict(dict)
            for candidate in candidates:
                for fold in range(plan.folds):
                    part = plan.partition(fold)
                    pick = lambda by_task, tasks: [r for t in sorted(tasks) for r in by_task.get(t, [])]

                    extra_source = set(by_task_s) - target_tasks
                    s_train = pick(by_task_s, set(part.train_tasks) | extra_source)
                    s_val, s_cal = pick(by_task_s, part.validation_tasks), pick(by_task_s, part.calibration_tasks)
                    t_train, t_val = pick(by_task_t, part.train_tasks), pick(by_task_t, part.validation_tasks)
                    t_cal, t_test = pick(by_task_t, part.calibration_tasks), pick(by_task_t, part.test_tasks)
                    if not t_test:
                        continue
                    common = dict(registry=registry, candidate=candidate, alpha=spec.alpha, calibrator_id=spec.calibrator_id, slice_=t_slice, test=t_test, weights=weights, seed=seed, fold=fold)
                    arms = {
                        "in_domain": dict(train=t_train, validation=t_val, calibration=t_cal),
                        "zero_shot": dict(train=s_train, validation=s_val, calibration=s_cal),
                        "calibration_transfer": dict(train=s_train, validation=s_val, calibration=t_cal),
                        "pooled": dict(train=s_train + t_train, validation=s_val + t_val, calibration=s_cal + t_cal),
                    }
                    train_task_list = sorted(part.train_tasks & target_tasks)
                    for k in shots:
                        for draw in range(DRAWS):
                            chosen = set(random.Random(seed * 1000 + fold * 10 + draw).sample(train_task_list, min(k, len(train_task_list))))
                            arms[f"few_shot_{k}:{draw}"] = dict(train=s_train + pick(by_task_t, chosen), validation=s_val + pick(by_task_t, chosen & set(part.validation_tasks)) or s_val, calibration=t_cal)
                    for name, arm in arms.items():
                        if not arm["train"] or not arm["validation"] or not arm["calibration"]:
                            continue
                        regimes[name].setdefault(candidate.candidate_id, {}).update(fit_predict(**common, **arm))
                print(f"{prefix} seed {seed} {candidate.candidate_id}: {len(regimes)} regimes", flush=True)
            block_scores: dict = defaultdict(dict)
            for name, by_candidate in regimes.items():
                for cid, forecasts in by_candidate.items():
                    rows = [r for r in t_rows if r.point.point_id in forecasts]
                    block_scores[cid][name] = results[prefix][cid][name][seed] = score(forecasts, rows, weights)
            (partial_dir / f"{prefix}--{seed}.json").write_text(json.dumps({"prefix": prefix, "seed": seed, "scores": block_scores}), encoding="utf-8")

    report = {"config_id": config.config_id, "source": str(args.source), "target": str(args.target), "shots": shots, "cells": {}}
    for prefix, by_candidate in results.items():
        report["cells"][prefix] = {}
        for cid, regimes in by_candidate.items():
            merged: dict[str, dict[int, dict]] = defaultdict(dict)
            for name, by_seed in regimes.items():
                base = name.split(":")[0]
                for seed, s in by_seed.items():
                    if base in merged and seed in merged[base]:
                        prev = merged[base][seed]
                        merged[base][seed] = {**{k: (prev[k] + s[k]) for k in ("mae", "wape", "coverage", "normalized_width")}, "task_mae": {t: prev["task_mae"][t] + s["task_mae"][t] for t in prev["task_mae"]}, "_n": prev["_n"] + 1}
                    else:
                        merged[base][seed] = {**{k: s[k] for k in ("mae", "wape", "coverage", "normalized_width")}, "task_mae": dict(s["task_mae"]), "_n": 1}
            summary = {}
            seed_task_mae: dict[str, dict[str, float]] = {}
            for base, by_seed in merged.items():
                for seed, s in by_seed.items():
                    n = s.pop("_n")
                    for k in ("mae", "wape", "coverage", "normalized_width"):
                        s[k] /= n
                    s["task_mae"] = {t: v / n for t, v in s["task_mae"].items()}
                summary[base] = {k: statistics.fmean(s[k] for s in by_seed.values()) for k in ("mae", "wape", "coverage", "normalized_width")}
                tasks = set.intersection(*(set(s["task_mae"]) for s in by_seed.values()))
                seed_task_mae[base] = {t: statistics.fmean(by_seed[seed]["task_mae"][t] for seed in by_seed) for t in tasks}
            for base in summary:
                if base != "in_domain":
                    summary[base]["vs_in_domain"] = paired(seed_task_mae[base], seed_task_mae["in_domain"])
                if base not in ("in_domain", "zero_shot") and "zero_shot" in seed_task_mae:
                    summary[base]["vs_zero_shot"] = paired(seed_task_mae[base], seed_task_mae["zero_shot"])
            report["cells"][prefix][cid] = summary
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "condition_transfer.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    lines = [f"# Condition transfer: {args.source.name} → {args.target.name} ({config.config_id})", "",
             "| position | candidate | regime | MAE | WAPE | coverage | norm. width | Δ vs in-domain [95% CI] | rel. | Δ vs zero-shot [95% CI] |", "|---|---|---|---:|---:|---:|---:|---|---:|---|"]
    for prefix, by_candidate in report["cells"].items():
        for cid, summary in by_candidate.items():
            for base, s in summary.items():
                d, z = s.get("vs_in_domain"), s.get("vs_zero_shot")
                dtxt = "—" if d is None else f"{d['delta']:+.4g} [{d['ci_lower']:+.4g}, {d['ci_upper']:+.4g}]"
                rel = "—" if d is None else f"{d['relative']:+.1%}"
                ztxt = "—" if z is None else f"{z['delta']:+.4g} [{z['ci_lower']:+.4g}, {z['ci_upper']:+.4g}]"
                lines.append(f"| {prefix} | {cid} | {base} | {s['mae']:.4g} | {s['wape']:.2%} | {s['coverage']:.3f} | {s['normalized_width']:.3g} | {dtxt} | {rel} | {ztxt} |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
