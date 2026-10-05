from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

METRIC_NAMES = ("mae", "coverage", "normalized_interval_width", "interval_score", "mean_extra_reserved_tokens")


def collect(grid: Path) -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(grid.glob("split-*/fit-*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for experiment_id, candidates in payload.items():
            experiment = experiment_id.split("--")[0]
            if "__" not in experiment:
                continue
            position, calibrator = experiment.split("__", 1)
            for candidate_id, record in candidates.items():
                if candidate_id == "empirical_quantile":
                    continue
                out[position][calibrator].append(record["metrics"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grid", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    collected = collect(args.grid)
    report = {
        position: {
            calibrator: {name: statistics.fmean(m[name] for m in metrics_list) for name in METRIC_NAMES}
            for calibrator, metrics_list in calibrators.items()
        }
        for position, calibrators in collected.items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "calibrator_summary.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    lines = ["# Calibrator comparison on the reported per-position point forecast", "",
             "MAE is identical across a position's calibrators by construction (only the interval changes); shown once for reference.", ""]
    for position, calibrators in report.items():
        lines += [f"## {position}", "", "| calibrator | MAE | coverage | normalized width | interval score | extra reserved |", "|---|---:|---:|---:|---:|---:|"]
        for calibrator, m in sorted(calibrators.items()):
            lines.append(f"| {calibrator} | {m['mae']:,.4g} | {m['coverage']:.3f} | {m['normalized_interval_width']:.3g} | {m['interval_score']:,.4g} | {m['mean_extra_reserved_tokens']:,.4g} |")
        lines.append("")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
