from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scripts.experiments.build_main_table import experiment_prefix, paired_bootstrap

SEED = 20260719


def read_rows(directory: Path) -> dict[str | None, list[dict]]:

    flat = directory / "predictions.jsonl"
    if flat.exists():
        return {None: [json.loads(line) for line in flat.open(encoding="utf-8")]}
    out = {}
    for fit_dir in sorted(directory.glob("split-*/fit-*")):
        out[fit_dir.parent.name] = [json.loads(line) for line in (fit_dir / "predictions.jsonl").open(encoding="utf-8")]
    if not out:
        raise SystemExit(f"no predictions under {directory}")
    return out


def score(rows: list[dict], weights: dict[str, float]) -> dict:

    err: dict[str, list[float]] = defaultdict(list)
    wsum: dict[str, float] = defaultdict(float)
    abs_total = target_total = covered = weight_total = 0.0
    for r in rows:
        w = weights[r["point_id"]]
        target, pred = float(r["target_value"]), float(r["prediction"])
        err[r["task_id"]].append(w * abs(pred - target))
        wsum[r["task_id"]] += w
        abs_total += w * abs(pred - target)
        target_total += w * target
        covered += w * float(float(r["lower"]) <= target <= float(r["upper"]))
        weight_total += w
    return {"task_mae": {t: sum(v) / wsum[t] for t, v in err.items()}, "wape": abs_total / target_total, "coverage": covered / weight_total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--arm-candidate", default="progressive_interval")
    parser.add_argument("--ours-rows", type=Path, action="append", required=True)
    parser.add_argument("--ours", action="append", required=True, help="<experiment_prefix>=<candidate_id>")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ours = dict(item.split("=", 1) for item in args.ours)
    arm_rows = read_rows(args.arm)
    ours_rows = [read_rows(d) for d in args.ours_rows]
    splits = sorted({s for src in ours_rows for s in src if s is not None} | {s for s in arm_rows if s is not None})
    if not splits:
        raise SystemExit("neither side has split directories")

    report = {"arm": str(args.arm), "arm_candidate": args.arm_candidate, "ours": ours, "splits": splits, "experiments": {}}
    for prefix, candidate in ours.items():
        per_seed_ours: dict[str, list[float]] = defaultdict(list)
        per_seed_arm: dict[str, list[float]] = defaultdict(list)
        wapes, points = [], set()
        for split in splits:
            mine = [r for src in ours_rows for r in src.get(split, src.get(None, [])) if r["candidate_id"] == candidate and experiment_prefix(r["experiment_id"]) == prefix and r.get("label_status", "observed") == "observed"]
            theirs = [r for r in arm_rows.get(split, arm_rows.get(None, [])) if r["candidate_id"] == args.arm_candidate and experiment_prefix(r["experiment_id"]) == prefix]
            shared = {r["point_id"] for r in mine} & {r["point_id"] for r in theirs}
            if not shared:
                continue
            mine = [r for r in mine if r["point_id"] in shared]
            theirs = [r for r in theirs if r["point_id"] in shared]
            weights = {r["point_id"]: float(r.get("sample_weight") or 1.0) for r in mine}
            a, b = score(mine, weights), score(theirs, weights)
            for t, v in a["task_mae"].items():
                per_seed_ours[t].append(v)
            for t, v in b["task_mae"].items():
                per_seed_arm[t].append(v)
            wapes.append((a["wape"], b["wape"], a["coverage"], b["coverage"]))
            points |= shared
        if not points:
            continue
        ours_task = {t: statistics.fmean(v) for t, v in per_seed_ours.items()}
        arm_task = {t: statistics.fmean(v) for t, v in per_seed_arm.items()}
        boot = paired_bootstrap(ours_task, arm_task, seed=SEED)
        report["experiments"][prefix] = {
            "points": len(points),
            "tasks": len(ours_task),
            "ours_mae": boot["mae_a"],
            "arm_mae": boot["mae_b"],
            "ours_wape": statistics.fmean(w[0] for w in wapes),
            "arm_wape": statistics.fmean(w[1] for w in wapes),
            "ours_coverage": statistics.fmean(w[2] for w in wapes),
            "arm_coverage": statistics.fmean(w[3] for w in wapes),
            "bootstrap": boot,
        }

    output = args.output or args.arm
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    lines = [
        f"# {args.arm_candidate} ({args.arm}) against ours on the same points",
        "",
        "| position | points | tasks | ours MAE | arm MAE | ours WAPE | arm WAPE | ours coverage | arm coverage | Δ ours − arm [95% CI] | task win |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for prefix, s in report["experiments"].items():
        b = s["bootstrap"]
        lines.append(
            f"| {prefix} | {s['points']} | {s['tasks']} | {s['ours_mae']:,.0f} | {s['arm_mae']:,.0f} | {s['ours_wape']:.3f} | {s['arm_wape']:.3f} "
            f"| {s['ours_coverage']:.3f} | {s['arm_coverage']:.3f} | {b['delta']:,.0f} [{b['ci_lower']:,.0f}, {b['ci_upper']:,.0f}] | {b['win_probability']:.2f} |"
        )
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
