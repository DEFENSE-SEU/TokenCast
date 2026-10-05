from __future__ import annotations

import argparse
import json
from pathlib import Path

CallKey = tuple[str, str]


def load_call_start(fit_dir: Path, candidate: str) -> dict[CallKey, dict]:
    by_call: dict[CallKey, dict] = {}
    with (fit_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["candidate_id"] != candidate or not record["experiment_id"].startswith("call_start"):
                continue
            by_call[(record["run_id"], record["logical_call_id"])] = record
    return by_call


def derive_split(fit_dir: Path, call_start_by_call: dict[CallKey, dict], in_call_candidate: str) -> tuple[list[dict], dict]:
    out_rows: list[dict] = []
    n_total = 0
    n_matched = 0
    with (fit_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["candidate_id"] != in_call_candidate or not record["experiment_id"].startswith("in_call_billed"):
                continue
            n_total += 1
            call_start = call_start_by_call.get((record["run_id"], record["logical_call_id"]))
            if call_start is None:
                continue
            n_matched += 1
            confirmed = float(record["progress_known_amount"])
            out_rows.append({
                "point_id": record["point_id"],
                "experiment_id": record["experiment_id"],
                "candidate_id": "call_start_static_ours",
                "run_id": record["run_id"],
                "task_id": record["task_id"],
                "fold": record["fold"],
                "target_value": record["target_value"],
                "sample_weight": record["sample_weight"],
                "prediction": max(float(call_start["prediction"]), confirmed),
                "lower": max(float(call_start["lower"]), confirmed),
                "upper": max(float(call_start["upper"]), confirmed),
                "label_status": "observed",
            })
    stats = {"n_points": n_total, "n_matched": n_matched, "match_rate": n_matched / n_total if n_total else float("nan")}
    return out_rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mt_grid", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--call-start-candidate", default="call_start_ours")
    parser.add_argument("--in-call-candidate", default="in_call_ours_clock")
    args = parser.parse_args()

    all_stats: dict[str, dict] = {}
    for split_dir in sorted(args.mt_grid.glob("split-*")):
        split_seed = split_dir.name.removeprefix("split-")
        fit_dirs = list(split_dir.glob("fit-*"))
        if not fit_dirs:
            continue
        call_start_by_call = load_call_start(fit_dirs[0], args.call_start_candidate)
        rows, stats = derive_split(fit_dirs[0], call_start_by_call, args.in_call_candidate)
        all_stats[split_seed] = stats
        out_dir = args.output / f"split-{split_seed}" / fit_dirs[0].name
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "derivation_summary.json").write_text(json.dumps(all_stats, indent=1), encoding="utf-8")
    lines = [
        "# Call Start static baseline (call_start_static_ours)",
        "",
        "prediction = max(Call Start prediction for the same (run_id, logical_call_id), amount already confirmed at the in-call checkpoint); lower/upper composed the same way.",
        "",
        "Join key is (run_id, logical_call_id) — Call Start predictions carry no attempt_id, so one Call Start prediction is shared by every attempt/retry of a call.",
        "",
        "| split seed | in-call points | matched to a Call Start prediction | match rate |",
        "|---|---:|---:|---:|",
    ]
    for seed, stats in sorted(all_stats.items()):
        lines.append(f"| {seed} | {stats['n_points']} | {stats['n_matched']} | {stats['match_rate']:.1%} |")
    lines.append("")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
