from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


CURRENT_CANDIDATES = {
    "call_start_ours": "call_start",
    "in_call_ours_clock": "in_call_billed",
    "task_update_ours": "task_update",
}


def collect(grid: Path) -> dict[str, dict[str, dict]]:


    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(grid.glob("split-*/fit-*/predictions.jsonl")):
        seed = path.parents[1].name.removeprefix("split-")
        counts: dict[str, dict[str, float]] = defaultdict(
            lambda: {"n_points": 0, "n_negative": 0, "most_negative": 0.0}
        )
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                position = CURRENT_CANDIDATES.get(row.get("candidate_id"))
                if position is None:
                    continue
                raw = row.get("raw_prediction")
                if raw is None:
                    continue
                stats = counts[position]
                stats["n_points"] += 1
                if raw < 0:
                    stats["n_negative"] += 1
                    stats["most_negative"] = min(stats["most_negative"], raw)
        for position, stats in counts.items():
            out[position][seed] = dict(stats)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("grid", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.grid / "ablation"

    report = collect(args.grid)
    if not report:
        raise SystemExit(f"no {sorted(CURRENT_CANDIDATES)} rows found under {args.grid}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "negative_raw_predictions.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8"
    )

    lines = [
        "# Points that would predict negative without the non-negative floor",
        "",
        "raw_prediction is the point estimate before "
        "LightGBMQuantileSession.predict's max(0.0, ...) floor.",
        "",
        "| position | seed | n_points | n_negative | negative_fraction | most_negative |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for position in sorted(report):
        for seed in sorted(report[position]):
            stats = report[position][seed]
            fraction = stats["n_negative"] / stats["n_points"] if stats["n_points"] else 0.0
            lines.append(
                f"| {position} | {seed} | {stats['n_points']} | {stats['n_negative']} | "
                f"{fraction:.4%} | {stats['most_negative']:.4g} |"
            )
    text = "\n".join(lines) + "\n"
    (output / "negative_raw_predictions.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
