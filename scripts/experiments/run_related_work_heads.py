from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

ALPHA = 0.10
FOLDS = 5


def load_grid_points(fit_dir: Path) -> dict[str, list[dict]]:

    chosen: dict[str, str] = {}
    rows: dict[str, list[dict]] = defaultdict(list)
    with (fit_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            experiment = record["experiment_id"]
            chosen.setdefault(experiment, record["candidate_id"])
            if record["candidate_id"] != chosen[experiment] or record.get("label_status") != "observed":
                continue
            rows[experiment].append({k: record[k] for k in ("point_id", "experiment_id", "run_id", "task_id", "fold", "target_value", "sample_weight", "cutoff_event_seq", "prediction_context_id", "logical_call_id", "source_event_id", "attempt_id", "checkpoint_ordinal", "known_offset_tokens", "progress_known_amount", "condition_id" if "condition_id" in record else "run_id")})
    return rows


def partition(fold: int) -> tuple[int, int]:
    return (fold + 1) % FOLDS, (fold + 2) % FOLDS


def bin_edges(y: np.ndarray, bins: int) -> np.ndarray:
    logs = np.log1p(y)
    edges = np.quantile(logs, np.linspace(0, 1, bins + 1)[1:-1])
    return np.unique(edges)


def fit_bin_classifier(x_tr, y_tr, w_tr, x_va, y_va, w_va, *, bins: int):
    edges = bin_edges(y_tr, bins)
    labels_tr = np.searchsorted(edges, np.log1p(y_tr))
    medians = np.array([np.median(y_tr[labels_tr == k]) if np.any(labels_tr == k) else float(np.expm1(edges[min(k, len(edges) - 1)])) for k in range(len(edges) + 1)])
    best = None
    for c in (0.01, 0.1, 1.0):
        model = LogisticRegression(C=c, max_iter=2000).fit(x_tr, labels_tr, sample_weight=w_tr)
        pred = predict_bins(model, medians, x_va)
        mae = np.average(np.abs(pred - y_va), weights=w_va)
        if best is None or mae < best[0]:
            best = (mae, model)
    return best[1], medians


def predict_bins(model, medians, x):
    probs = model.predict_proba(x)
    values = medians[model.classes_]
    return probs @ values


def fit_ridge_log(x_tr, y_tr, w_tr, x_va, y_va, w_va):
    best = None
    for alpha in (1.0, 10.0, 100.0, 1000.0):
        model = Ridge(alpha=alpha).fit(x_tr, np.log1p(y_tr), sample_weight=w_tr)
        pred = np.expm1(model.predict(x_va)).clip(min=0)
        mae = np.average(np.abs(pred - y_va), weights=w_va)
        if best is None or mae < best[0]:
            best = (mae, model)
    return best[1]


def conformal_radius(residuals: np.ndarray, weights: np.ndarray, coverage: float = 1 - ALPHA) -> float:
    order = np.argsort(residuals)
    cum = np.cumsum(weights[order]) / weights.sum()
    index = np.searchsorted(cum, coverage)
    return float(residuals[order][min(index, len(residuals) - 1)])

def calibrate_in_fold(rows: list[dict], raw: dict[str, float]) -> dict[str, dict[str, tuple[float, float, float]]]:

    fold_of = {r["task_id"]: int(r["fold"]) for r in rows}
    out: dict[str, dict[str, tuple[float, float, float]]] = {"raw": {}, "calibrated": {}}
    for fold in range(FOLDS):
        cal_fold, val_fold = partition(fold)
        train = [r for r in rows if fold_of[r["task_id"]] not in (fold, cal_fold, val_fold)]
        cal = [r for r in rows if fold_of[r["task_id"]] == cal_fold]
        test = [r for r in rows if fold_of[r["task_id"]] == fold]
        if not train or not cal or not test:
            continue
        x = np.log1p([raw[r["point_id"]] for r in train])
        y = np.log1p([float(r["target_value"]) for r in train])
        w = np.array([float(r["sample_weight"]) for r in train])
        a, b = np.polyfit(x, y, 1, w=np.sqrt(w)) if len(set(x.tolist())) > 1 else (1.0, float(np.average(y - x, weights=w)))
        transforms = {"raw": lambda v: v, "calibrated": lambda v: float(np.expm1(a * math.log1p(v) + b))}
        for name, transform in transforms.items():
            residuals = np.array([abs(transform(raw[r["point_id"]]) - float(r["target_value"])) for r in cal])
            radius = conformal_radius(residuals, np.array([float(r["sample_weight"]) for r in cal]))
            for r in test:
                value = transform(raw[r["point_id"]])
                out[name][r["point_id"]] = (value, max(0.0, value - radius), value + radius)
    return out


def fit_bin_classifier_uniform(x_tr, y_tr, w_tr, x_va, y_va, w_va, *, bins: int):

    edges = np.linspace(float(y_tr.min()), float(y_tr.max()), bins + 1)[1:-1]
    labels_tr = np.searchsorted(edges, y_tr)
    medians = np.array([np.median(y_tr[labels_tr == k]) if np.any(labels_tr == k) else float(edges[min(k, len(edges) - 1)]) for k in range(len(edges) + 1)])
    best = None
    for c in (0.01, 0.1, 1.0):
        model = LogisticRegression(C=c, max_iter=2000).fit(x_tr, labels_tr, sample_weight=w_tr)
        mae = np.average(np.abs(predict_bins(model, medians, x_va) - y_va), weights=w_va)
        if best is None or mae < best[0]:
            best = (mae, model)
    return best[1], medians


def fit_forest(x_tr, y_tr, w_tr, x_va, y_va, w_va):

    best = None
    for depth in (6, 12, None):
        model = RandomForestRegressor(n_estimators=200, max_depth=depth, random_state=0, n_jobs=2).fit(x_tr, np.log1p(y_tr), sample_weight=w_tr)
        mae = np.average(np.abs(np.expm1(model.predict(x_va)).clip(min=0) - y_va), weights=w_va)
        if best is None or mae < best[0]:
            best = (mae, model)
    return best[1]


def fit_rank(x_tr, y_tr, w_tr, x_va, y_va, w_va):

    order = np.argsort(y_tr)
    ranks = np.empty(len(y_tr))
    ranks[order] = np.linspace(0.0, 1.0, len(y_tr))
    quantiles = np.sort(y_tr)
    best = None
    for alpha in (1.0, 10.0, 100.0):
        model = Ridge(alpha=alpha).fit(x_tr, ranks, sample_weight=w_tr)
        pred = quantiles[np.clip((model.predict(x_va) * (len(quantiles) - 1)).round().astype(int), 0, len(quantiles) - 1)]
        mae = np.average(np.abs(pred - y_va), weights=w_va)
        if best is None or mae < best[0]:
            best = (mae, model)
    return best[1], quantiles


def rank_to_value(model, quantiles, x):
    return quantiles[np.clip((model.predict(x) * (len(quantiles) - 1)).round().astype(int), 0, len(quantiles) - 1)]


def fit_progress(x_tr, y_tr, w_tr, offsets_tr):

    p_tr = np.clip(offsets_tr / np.maximum(offsets_tr + y_tr, 1e-6), 1e-3, 1 - 1e-3)
    model = Ridge(alpha=10.0).fit(x_tr, p_tr, sample_weight=w_tr)
    return model, (float(np.quantile(p_tr, 0.01)), float(np.quantile(p_tr, 0.99)))


def progress_to_remaining(fitted, x, offsets):
    model, (low, high) = fitted
    p = np.clip(model.predict(x), low, high)
    return (offsets * (1.0 - p) / p).clip(min=0)


def fit_predict_head(name, xs, y, w, offsets, tr, va, ca, te, *, bins: int, nu: float, levels: tuple[float, ...] = ()):

    te_mu = te_sigma = None
    if name in ("trail_last", "trail_mean", "s3_encoder") or name.startswith("trail_last_layer"):
        model, medians = fit_bin_classifier(xs[tr], y[tr], w[tr], xs[va], y[va], w[va], bins=bins)
        point = lambda m: predict_bins(model, medians, m)
        cal_pred, t0 = point(xs[ca]), time.perf_counter()
        te_pred = point(xs[te])
        ms = (time.perf_counter() - t0) * 1000 / te.sum()
        radius = conformal_radius(np.abs(cal_pred - y[ca]), w[ca])
        lo, hi = np.clip(te_pred - radius, 0, None), te_pred + radius
    elif name in ("egtp_pool", "ssjf_reg", "estp"):
        model = fit_ridge_log(xs[tr], y[tr], w[tr], xs[va], y[va], w[va])
        cal_pred = np.expm1(model.predict(xs[ca])).clip(min=0)
        t0 = time.perf_counter()
        te_pred = np.expm1(model.predict(xs[te])).clip(min=0)
        ms = (time.perf_counter() - t0) * 1000 / te.sum()
        radius = conformal_radius(np.abs(cal_pred - y[ca]), w[ca])
        lo, hi = np.clip(te_pred - radius, 0, None), te_pred + radius
    elif name == "ssjf_mc":
        model, medians = fit_bin_classifier_uniform(xs[tr], y[tr], w[tr], xs[va], y[va], w[va], bins=bins)
        cal_pred, t0 = predict_bins(model, medians, xs[ca]), time.perf_counter()
        te_pred = predict_bins(model, medians, xs[te])
        ms = (time.perf_counter() - t0) * 1000 / te.sum()
        radius = conformal_radius(np.abs(cal_pred - y[ca]), w[ca])
        lo, hi = np.clip(te_pred - radius, 0, None), te_pred + radius
    elif name == "magnus":
        model = fit_forest(xs[tr], y[tr], w[tr], xs[va], y[va], w[va])
        cal_pred = np.expm1(model.predict(xs[ca])).clip(min=0)
        t0 = time.perf_counter()
        te_pred = np.expm1(model.predict(xs[te])).clip(min=0)
        ms = (time.perf_counter() - t0) * 1000 / te.sum()
        radius = conformal_radius(np.abs(cal_pred - y[ca]), w[ca])
        lo, hi = np.clip(te_pred - radius, 0, None), te_pred + radius
    elif name == "ltr_c":
        model, quantiles = fit_rank(xs[tr], y[tr], w[tr], xs[va], y[va], w[va])
        cal_pred = rank_to_value(model, quantiles, xs[ca])
        t0 = time.perf_counter()
        te_pred = rank_to_value(model, quantiles, xs[te])
        ms = (time.perf_counter() - t0) * 1000 / te.sum()
        radius = conformal_radius(np.abs(cal_pred - y[ca]), w[ca])
        lo, hi = np.clip(te_pred - radius, 0, None), te_pred + radius
    elif name == "tpv":
        model = fit_progress(xs[tr], y[tr], w[tr], offsets[tr])
        cal_pred = progress_to_remaining(model, xs[ca], offsets[ca])
        t0 = time.perf_counter()
        te_pred = progress_to_remaining(model, xs[te], offsets[te])
        ms = (time.perf_counter() - t0) * 1000 / te.sum()
        radius = conformal_radius(np.abs(cal_pred - y[ca]), w[ca])
        lo, hi = np.clip(te_pred - radius, 0, None), te_pred + radius
    else:
        mu_model = fit_ridge_log(xs[tr], y[tr], w[tr], xs[va], y[va], w[va])
        resid_tr = np.log1p(y[tr]) - mu_model.predict(xs[tr])
        sigma_model = Ridge(alpha=100.0).fit(xs[tr], np.log1p(np.abs(resid_tr)), sample_weight=w[tr])
        q = student_t.ppf(1 - ALPHA / 2, nu)
        def quantiles(m):
            mu = mu_model.predict(m)
            sigma = np.expm1(sigma_model.predict(m)).clip(min=1e-3)
            return np.expm1(mu).clip(min=0), np.expm1(mu - q * sigma).clip(min=0), np.expm1(mu + q * sigma)
        cal_point, cal_lo, cal_hi = quantiles(xs[ca])

        ratios = np.maximum((cal_point - y[ca]) / np.maximum(cal_point - cal_lo, 1e-6), (y[ca] - cal_point) / np.maximum(cal_hi - cal_point, 1e-6))
        factor = max(1.0, conformal_radius(ratios, w[ca]))
        t0 = time.perf_counter()
        te_pred, te_lo, te_hi = quantiles(xs[te])
        ms = (time.perf_counter() - t0) * 1000 / te.sum()
        lo, hi = np.clip(te_pred - factor * (te_pred - te_lo), 0, None), te_pred + factor * (te_hi - te_pred)
        te_mu, te_sigma = mu_model.predict(xs[te]), np.expm1(sigma_model.predict(xs[te])).clip(min=1e-3)
    level_values = {}
    for level in levels:
        if level == 0.5:
            level_values[level] = te_pred
        elif te_mu is not None:
            z = student_t.ppf(level, nu) if level > 0.5 else -student_t.ppf(1 - level, nu)
            base = np.expm1(te_mu + z * te_sigma)
            level_values[level] = np.clip(te_pred - factor * (te_pred - base.clip(min=0)), 0, None) if level < 0.5 else te_pred + factor * (base - te_pred)
        else:
            radius = conformal_radius(np.abs(cal_pred - y[ca]), w[ca], round(abs(2 * level - 1), 10))
            level_values[level] = np.clip(te_pred - radius, 0, None) if level < 0.5 else te_pred + radius
    return te_pred, lo, hi, ms, level_values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grid", type=Path)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--nu", type=float, default=3.5)
    parser.add_argument("--heads", help="comma-separated head names to fit (default: all)")
    parser.add_argument("--positions", help="comma-separated experiment prefixes to fit, e.g. task_update (default: all)")
    parser.add_argument("--quantiles", help="comma-separated levels written to each row's quantiles field, e.g. 0.05,0.1,...,0.95")
    args = parser.parse_args()
    levels = tuple(float(q) for q in args.quantiles.split(",")) if args.quantiles else ()
    reps = np.load(args.representations / "representations.npz")
    index = {pid: i for i, pid in enumerate(reps["point_id"])}
    feature_sets = {
        "trail_last": lambda idx: reps["proxy_last"][idx].astype(np.float32),
        "trail_mean": lambda idx: reps["proxy_mean"][idx].astype(np.float32),
        "egtp_pool": lambda idx: np.concatenate([reps["proxy_entropy_pool"][idx].astype(np.float32), reps["entropy_stats"][idx]], axis=1),
        "s3_encoder": lambda idx: reps["encoder_cls"][idx].astype(np.float32),
        "tie_encoder": lambda idx: np.concatenate([reps["encoder_cls"][idx], reps["encoder_mean"][idx], reps["encoder_max"][idx]], axis=1).astype(np.float32),
        "ssjf_reg": lambda idx: reps["encoder_cls"][idx].astype(np.float32),
        "ssjf_mc": lambda idx: reps["encoder_cls"][idx].astype(np.float32),
        "magnus": lambda idx: np.concatenate([reps["encoder_cls"][idx].astype(np.float32), np.log1p(reps["entropy_stats"][idx][:, 2:3])], axis=1),
        "ltr_c": lambda idx: reps["encoder_cls"][idx].astype(np.float32),
        "estp": lambda idx: np.concatenate([reps["proxy_entropy_pool"][idx].astype(np.float32), reps["proxy_mean"][idx].astype(np.float32), reps["entropy_stats"][idx]], axis=1),
        "tpv": lambda idx: reps["proxy_last"][idx].astype(np.float32),
    }


    for key in sorted(k for k in reps.files if k.startswith("proxy_last_layer")):
        feature_sets[f"trail_last_{key.rsplit('_', 1)[-1]}"] = (lambda k: lambda idx: reps[k][idx].astype(np.float32))(key)
    if args.heads:
        feature_sets = {name: feature_sets[name] for name in args.heads.split(",")}

    needs_offset = {"tpv"}
    summary_rows = []
    for fit_dir in sorted(args.grid.glob("split-*/fit-*")):
        out_dir = args.output / fit_dir.parent.name / fit_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        sink = (out_dir / "predictions.jsonl").open("w", encoding="utf-8")
        for experiment, rows in load_grid_points(fit_dir).items():
            if args.positions and experiment.split("--")[0] not in args.positions.split(","):
                continue
            rows = [r for r in rows if r["point_id"] in index]
            if not rows:
                continue
            fold_of = {r["task_id"]: int(r["fold"]) for r in rows}
            idx = np.array([index[r["point_id"]] for r in rows])
            y = np.array([float(r["target_value"]) for r in rows])
            w = np.array([float(r["sample_weight"]) for r in rows])
            folds = np.array([fold_of[r["task_id"]] for r in rows])


            offsets = np.array([float(r.get("known_offset_tokens") or 0.0) for r in rows])
            for name, extract in feature_sets.items():
                if name in needs_offset and not (offsets > 0).all():
                    continue
                x_all = extract(idx)
                predictions = np.zeros(len(rows))
                lower = np.zeros(len(rows))
                upper = np.zeros(len(rows))
                latency = np.zeros(len(rows))
                quantiles = {level: np.zeros(len(rows)) for level in levels}
                for fold in range(FOLDS):
                    cal_fold, val_fold = partition(fold)
                    tr = ~np.isin(folds, [fold, cal_fold, val_fold])
                    va, ca, te = folds == val_fold, folds == cal_fold, folds == fold
                    if te.sum() == 0 or tr.sum() < 10:
                        continue
                    scaler = StandardScaler().fit(x_all[tr])
                    xs = scaler.transform(x_all)
                    te_pred, lo, hi, ms, te_quantiles = fit_predict_head(name, xs, y, w, offsets, tr, va, ca, te, bins=args.bins, nu=args.nu, levels=levels)
                    predictions[te], lower[te], upper[te], latency[te] = te_pred, lo, hi, ms
                    for level, values in te_quantiles.items():
                        quantiles[level][te] = values
                for i, r in enumerate(rows):
                    extra = {"quantiles": {str(level): float(quantiles[level][i]) for level in levels}} if levels else {}
                    sink.write(json.dumps({**{k: r[k] for k in ("point_id", "experiment_id", "run_id", "task_id", "fold", "target_value", "sample_weight")}, "candidate_id": name, "prediction": float(predictions[i]), "lower": float(lower[i]), "upper": float(upper[i]), "label_status": "observed", "latency_ms": float(latency[i]), **extra}) + "\n")
                by_task: dict[str, list] = defaultdict(list)
                for i, r in enumerate(rows):
                    by_task[r["task_id"]].append((w[i], abs(predictions[i] - y[i]), float(lower[i] <= y[i] <= upper[i]), (upper[i] - lower[i]) / max(y[i], 1.0)))
                task_mae = {t: sum(a * b for a, b, _, _ in v) / sum(a for a, *_ in v) for t, v in by_task.items()}
                summary_rows.append({"split": fit_dir.parent.name, "experiment": experiment.split("--")[0], "candidate": name, "mae": statistics.fmean(task_mae.values()), "wape": float(np.sum(w * np.abs(predictions - y)) / np.sum(w * y)), "coverage": statistics.fmean(sum(a * c for a, _, c, _ in v) / sum(a for a, *_ in v) for v in by_task.values()), "normalized_width": statistics.fmean(sum(a * d for a, _, _, d in v) / sum(a for a, *_ in v) for v in by_task.values()), "n_points": len(rows), "n_tasks": len(by_task)})
                print(f"{fit_dir.parent.name} {experiment.split('--')[0]} {name}: mae {summary_rows[-1]['mae']:.4g} wape {summary_rows[-1]['wape']:.4f} cov {summary_rows[-1]['coverage']:.3f}", flush=True)
        sink.close()
    grouped: dict[tuple, list] = defaultdict(list)
    for r in summary_rows:
        grouped[(r["experiment"], r["candidate"])].append(r)
    lines = ["# Related-work heads on proxy representations (seed means)", "", "| position | head | MAE | WAPE | coverage | norm. width | points | tasks | seeds |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for (experiment, candidate), items in sorted(grouped.items()):
        lines.append(f"| {experiment} | {candidate} | {statistics.fmean(i['mae'] for i in items):.4g} | {statistics.fmean(i['wape'] for i in items):.4f} | {statistics.fmean(i['coverage'] for i in items):.3f} | {statistics.fmean(i['normalized_width'] for i in items):.3g} | {items[0]['n_points']} | {items[0]['n_tasks']} | {len(items)} |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary_rows, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
