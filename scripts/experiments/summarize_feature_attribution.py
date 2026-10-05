from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_CANDIDATES = {
    "call_start": "call_start_ours",
    "in_call_visible": "in_call_ours",
    "in_call_billed": "in_call_ours_clock",
    "task_update": "task_update_ours",
}
MEDIAN_QUANTILE = 0.5


def subgroup_of(feature: str) -> str:

    from token_prediction.features.selection import DEFAULT_FEATURE_CATALOG

    spec = DEFAULT_FEATURE_CATALOG.get(feature)
    return spec.subgroup if spec is not None else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grid", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], help="<position>=<candidate_id>")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    wanted = dict(DEFAULT_CANDIDATES)
    wanted.update(dict(item.split("=", 1) for item in args.candidate))

    per_position: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    folds_seen: dict[str, int] = defaultdict(int)
    for path in sorted(args.grid.glob("split-*/fit-*/models/*/*/seed-*/fold-*/feature-importance.json")):
        candidate = path.parents[2].name
        position = path.parents[3].name.split("-", 2)[-1]
        if wanted.get(position) != candidate:
            continue
        records = [r for r in json.loads(path.read_text(encoding="utf-8")) if abs(float(r["quantile"]) - MEDIAN_QUANTILE) < 1e-9]
        total = sum(float(r["gain"]) for r in records)
        if total <= 0:
            continue
        folds_seen[position] += 1
        for record in records:
            per_position[position][record["source_feature_name"]].append(float(record["gain"]) / total)

    report: dict = {"grid": str(args.grid), "candidates": wanted, "positions": {}}
    for position, features in per_position.items():
        n_folds = folds_seen[position]
        shares = {name: sum(values) / n_folds for name, values in features.items()}
        groups: dict[str, float] = defaultdict(float)
        for name, share in shares.items():
            groups[subgroup_of(name)] += share
        report["positions"][position] = {
            "n_folds": n_folds,
            "features": dict(sorted(shares.items(), key=lambda kv: -kv[1])),
            "subgroups": dict(sorted(groups.items(), key=lambda kv: -kv[1])),
            "features_used": sum(1 for v in shares.values() if v > 0),
        }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "feature_attribution.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    lines = [f"# What the fitted models read ({args.grid}, median quantile, gain share per fold)", ""]
    for position, block in report["positions"].items():
        lines += [f"## {position} (`{wanted[position]}`, {block['n_folds']} folds, {block['features_used']} features with a split)", "",
                  "| subgroup | gain share |", "|---|---:|"]
        for group, share in list(block["subgroups"].items())[: args.top]:
            if share > 0.001:
                lines.append(f"| {group} | {share:.1%} |")
        lines += ["", "| feature | gain share |", "|---|---:|"]
        for name, share in list(block["features"].items())[: args.top]:
            if share > 0.001:
                lines.append(f"| {name} | {share:.1%} |")
        lines.append("")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
