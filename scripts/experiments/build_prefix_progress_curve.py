from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scripts.experiments.build_main_table import experiment_prefix


def load(fit_dir: Path, weights: dict[str, float] | None = None) -> list[dict]:
    rows = []
    with (fit_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("label_status", "observed") != "observed":
                continue
            weight = record.get("sample_weight")
            if weight is None:
                weight = weights.get(record["point_id"]) if weights else None
                if weight is None:
                    continue
            rows.append({
                "point_id": record["point_id"],
                "experiment": experiment_prefix(record["experiment_id"]),
                "candidate": record["candidate_id"],
                "task_id": record["task_id"],
                "prediction": float(record["prediction"]),
                "target": float(record["target_value"]),
                "weight": float(weight),
                "produced": float(record.get("known_offset_tokens") or record.get("progress_known_amount") or 0.0),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grid", type=Path)
    parser.add_argument("--extra", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", default="in_call_visible")
    parser.add_argument("--buckets", type=int, default=5)
    parser.add_argument("--limit-points-from", type=Path, default=None)
    args = parser.parse_args()

    fit_dirs = sorted(args.grid.glob("split-*/fit-*"))
    if not fit_dirs:
        raise SystemExit(f"no split-*/fit-* under {args.grid}")
    per_seed: list[dict] = []
    shares: dict[str, float] = {}
    for fit_dir in fit_dirs:
        rows = [r for r in load(fit_dir) if r["experiment"] == args.experiment]
        weights = {r["point_id"]: r["weight"] for r in rows}
        allowed_ids: set[str] | None = None
        if args.limit_points_from is not None:
            limit_dirs = list((args.limit_points_from / fit_dir.parent.name).glob("fit-*")) or (
                [args.limit_points_from] if (args.limit_points_from / "predictions.jsonl").is_file() else []
            )
            if not limit_dirs:
                raise SystemExit(f"--limit-points-from has no split matching {fit_dir.parent.name}")
            allowed_ids = {r["point_id"] for r in load(limit_dirs[0], weights) if r["experiment"] == args.experiment}


        for r in rows:
            if allowed_ids is not None and r["point_id"] not in allowed_ids:
                continue
            total = r["produced"] + r["target"]
            if total > 0:
                shares[r["point_id"]] = r["produced"] / total
        for extra in args.extra:
            extra_dirs = list((extra / fit_dir.parent.name).glob("fit-*")) or ([extra] if (extra / "predictions.jsonl").is_file() else [])
            if extra_dirs:
                rows += [r for r in load(extra_dirs[0], weights) if r["experiment"] == args.experiment]
        per_seed.append(rows)
    if not shares:
        raise SystemExit(f"{args.experiment} carries no produced-so-far amount")


    order = sorted(shares, key=lambda pid: (shares[pid], pid))
    band_of = {pid: min(args.buckets - 1, index * args.buckets // len(order)) for index, pid in enumerate(order)}
    band_share: dict[int, list[float]] = defaultdict(list)
    for pid, share in shares.items():
        band_share[band_of[pid]].append(share)


    per_seed_band: list[dict] = []
    for rows in per_seed:
        acc: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0.0, 0.0])))
        for r in rows:
            band = band_of.get(r["point_id"])
            if band is None:
                continue
            cell = acc[r["candidate"]][band][r["task_id"]]
            cell[0] += r["weight"] * abs(r["prediction"] - r["target"])
            cell[1] += r["weight"]
        per_seed_band.append({c: {b: statistics.fmean(v[0] / v[1] for v in tasks.values()) for b, tasks in bands.items()} for c, bands in acc.items()})

    candidates = sorted(set.intersection(*(set(seed) for seed in per_seed_band)))
    report = {
        "grid": str(args.grid), "experiment": args.experiment, "n_seeds": len(fit_dirs),
        "bands": [{"band": b, "produced_share_median": statistics.median(v), "n_points": len(v)} for b, v in sorted(band_share.items())],
        "mae": {c: {str(b): statistics.fmean(seed[c][b] for seed in per_seed_band if b in seed[c]) for b in sorted(band_share)} for c in candidates},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "prefix_progress.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    band_labels = [f"{b['produced_share_median']:.0%}" for b in report["bands"]]
    lines = [f"# In-call error by the share of the Call already produced ({args.experiment}, {report['n_seeds']} seeds)", "",
             "Bands are equal-count over the produced share; the column header is the band's median share.",
             f"Each band holds about {report['bands'][0]['n_points']} points.", "",
             "| candidate | " + " | ".join(band_labels) + " | last − first |", "|---|" + "---:|" * (len(band_labels) + 1)]
    for candidate in sorted(candidates, key=lambda c: statistics.fmean(report["mae"][c].values())):
        values = [report["mae"][candidate][str(b["band"])] for b in report["bands"]]
        drop = values[-1] - values[0]
        lines.append(f"| {candidate} | " + " | ".join(f"{v:,.4g}" for v in values) + f" | {drop:+,.4g} |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
