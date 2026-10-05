from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import roc_auc_score


def load_experiment(fit_dir: Path, candidate: str) -> list[dict]:
    rows = []
    with (fit_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["candidate_id"] != candidate or not record["experiment_id"].startswith("in_call_billed"):
                continue
            rows.append({
                "point_id": record["point_id"], "task_id": record["task_id"],
                "target": float(record["target_value"]), "prediction": float(record["prediction"]),
                "weight": float(record["sample_weight"]),
            })
    return rows


def weighted_auc(rows: list[dict], threshold: float) -> float | None:
    y_true = [1 if r["target"] > threshold else 0 for r in rows]
    if len(set(y_true)) < 2:
        return None
    y_score = [r["prediction"] for r in rows]
    weight = [r["weight"] for r in rows]
    return float(roc_auc_score(y_true, y_score, sample_weight=weight))


def bucket_by_rank(rows: list[dict], buckets: int) -> dict[str, int]:
    order = sorted({r["point_id"]: r["target"] for r in rows}.items(), key=lambda kv: (kv[1], kv[0]))
    return {pid: min(buckets - 1, index * buckets // len(order)) for index, (pid, _) in enumerate(order)}


def weighted_mae_by_bucket(rows: list[dict], band_of: dict[str, int]) -> dict[int, float]:
    acc: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for r in rows:
        band = band_of.get(r["point_id"])
        if band is None:
            continue
        cell = acc[band][r["task_id"]]
        cell[0] += r["weight"] * abs(r["prediction"] - r["target"])
        cell[1] += r["weight"]
    return {band: statistics.fmean(v[0] / v[1] for v in tasks.values()) for band, tasks in acc.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("state_grid", type=Path)
    parser.add_argument("closure_grid", type=Path)
    parser.add_argument("full_grid", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-candidate", default="ic_billed_without_generation_prefix")
    parser.add_argument("--closure-candidate", default="ic_billed_with_closure_no_embedding")
    parser.add_argument("--full-candidate", default="in_call_ours_clock")
    parser.add_argument("--threshold", type=float, default=2048.0)
    parser.add_argument("--buckets", type=int, default=5)
    args = parser.parse_args()

    variants = {"state_only": (args.state_grid, args.state_candidate), "closure": (args.closure_grid, args.closure_candidate), "full": (args.full_grid, args.full_candidate)}
    full_fit_dirs = sorted(args.full_grid.glob("split-*/fit-*"))
    if not full_fit_dirs:
        raise SystemExit(f"no split-*/fit-* under {args.full_grid}")

    auc_by_variant: dict[str, list[float]] = defaultdict(list)
    mae_by_variant: dict[str, list[dict[int, float]]] = defaultdict(list)
    bucket_median: dict[int, list[float]] = defaultdict(list)
    for full_fit_dir in full_fit_dirs:
        split_name = full_fit_dir.parent.name
        full_rows = load_experiment(full_fit_dir, args.full_candidate)
        band_of = bucket_by_rank(full_rows, args.buckets)
        for band, target in ((band_of[r["point_id"]], r["target"]) for r in full_rows):
            bucket_median[band].append(target)
        for name, (grid, candidate) in variants.items():
            fit_dirs = list((grid / split_name).glob("fit-*"))
            if not fit_dirs:
                raise SystemExit(f"{grid} has no {split_name}")
            rows = load_experiment(fit_dirs[0], candidate)
            auc = weighted_auc(rows, args.threshold)
            if auc is not None:
                auc_by_variant[name].append(auc)
            mae_by_variant[name].append(weighted_mae_by_bucket(rows, band_of))

    report = {
        "threshold": args.threshold,
        "buckets": [{"band": b, "target_median": statistics.median(v), "n_points": len(v)} for b, v in sorted(bucket_median.items())],
        "auc": {name: {"mean": statistics.fmean(values), "per_seed": values} for name, values in auc_by_variant.items()},
        "mae_by_bucket": {name: {str(b): statistics.fmean(seed[b] for seed in seeds if b in seed) for b in sorted(bucket_median)} for name, seeds in mae_by_variant.items()},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "generation_prefix_mechanism.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    lines = [
        "# Generation-prefix mechanism: discrimination and error by remaining size", "",
        f"Threshold: remaining > {args.threshold:.0f} tokens. AUC uses each candidate's own point prediction as the ranking score, sample-weighted, averaged over {len(full_fit_dirs)} split seeds.", "",
        "| variant | AUC (mean over seeds) | per-seed AUC |", "|---|---:|---|",
    ]
    for name, stats in report["auc"].items():
        lines.append(f"| {name} | {stats['mean']:.3f} | " + ", ".join(f"{v:.3f}" for v in stats["per_seed"]) + " |")
    lines += ["", "Weighted MAE by remaining-size bucket (equal-count bands over the full variant's own target rank):", "",
              "| variant | " + " | ".join(f"{b['target_median']:.0f}" for b in report["buckets"]) + " |",
              "|---|" + "---:|" * len(report["buckets"])]
    for name, values in report["mae_by_bucket"].items():
        lines.append(f"| {name} | " + " | ".join(f"{values.get(str(b['band']), float('nan')):.3g}" for b in report["buckets"]) + " |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
