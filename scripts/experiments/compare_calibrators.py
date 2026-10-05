from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "workspace" / "runs"
POSITIONS = ["task_initialization", "call_start", "in_call_visible", "in_call_billed", "task_update"]
TITLES = {"task_initialization": "Task launch", "call_start": "Call start",
          "in_call_visible": "In-call visible", "in_call_billed": "In-call billed",
          "task_update": "Task update"}
DEFAULT_ARMS = {
    "magnitude bin": "sub-magnitude-frontier",
    "point width-scaled": "sub-cal-width-frontier",
    "task max": "sub-cal-taskmax-frontier",
}


def read_frontier(directory: Path) -> dict:

    path = directory / "interval_frontier.json"
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for cell, arms in report.get("experiments", {}).items():
        position = cell.split("--")[0]
        for candidate, block in arms.items():
            out.setdefault(position, {})[candidate] = {
                "width": block.get("width_at_matched_coverage"),
                "reserved": block.get("reserved_at_matched_risk"),
                "width_extrapolated": block.get("width_extrapolated"),
                "reserve_extrapolated": block.get("reserve_extrapolated"),
                "alphas": sorted(block.get("per_alpha", {})),
            }
    return out


def reported_arms(table: Path) -> dict[str, str]:

    arms: dict[str, str] = {}
    for line in table.read_text(encoding="utf-8").split("\n"):
        if not line.startswith("### "):
            continue
        head, _, rest = line[4:].partition("(ours = `")
        candidate, _, _ = rest.partition("`")
        if candidate:
            arms[head.strip()] = candidate
    if not arms:
        raise ValueError(f"no reported arms found in {table}")
    return arms


def pick(frontier: dict, position: str, prefer: str = "") -> tuple[str, dict] | tuple[None, None]:

    arms = frontier.get(position) or {}
    if not arms:
        return None, None
    if prefer:
        if prefer not in arms:
            return None, None
        return prefer, arms[prefer]
    candidate = next(iter(arms))
    return candidate, arms[candidate]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--table", type=Path, default=RUNS / "mt-v18" / "table" / "summary.md",
                        help="main table whose headers name the reported arm at each position")
    parser.add_argument("--arm", action="append", default=[], help="<name>=<frontier dir under workspace/runs>")
    args = parser.parse_args()

    arms = dict(DEFAULT_ARMS)
    arms.update(dict(item.split("=", 1) for item in args.arm))
    reference_dirs = [args.reference] if args.reference else [
        RUNS / "mt-v18" / "frontier-cs-ic", RUNS / "mt-v18" / "frontier-tl-tu"]
    reference: dict = {}
    for directory in reference_dirs:
        for position, block in read_frontier(Path(directory)).items():
            reference.setdefault(position, {}).update(block)

    reported = reported_arms(args.table)
    report: dict = {"reference": [str(d) for d in reference_dirs], "table": str(args.table),
                    "reported_arms": reported, "arms": {}, "positions": {}, "unscored": {}}
    loaded = {name: read_frontier(RUNS / path) for name, path in arms.items()}
    for name, path in arms.items():
        report["arms"][name] = {"path": str(RUNS / path), "present": bool(loaded[name])}

    dominating: list[str] = []
    for position in POSITIONS:
        wanted = reported.get(position)
        ours_id, ours = pick(reference, position, prefer=wanted or "")
        if ours is None or ours["width"] is None:
            report["unscored"][position] = (
                f"the reported arm `{wanted}` has no alpha sweep in the reference frontier"
                if wanted else "the main table names no reported arm here")
            continue
        cell: dict = {"reported_arm": ours_id, "reported": ours, "alternatives": {}}
        for name, frontier in loaded.items():
            candidate, block = pick(frontier, position)
            if block is None or block["width"] is None:
                cell["alternatives"][name] = {"present": False}
                continue
            narrower = block["width"] <= ours["width"]
            reserves_less = (block["reserved"] is not None and ours["reserved"] is not None
                             and block["reserved"] <= ours["reserved"])
            cell["alternatives"][name] = {
                "present": True, "candidate": candidate,
                "width": block["width"], "reserved": block["reserved"],
                "width_ratio": block["width"] / ours["width"] if ours["width"] else None,
                "reserved_ratio": (block["reserved"] / ours["reserved"]
                                   if ours["reserved"] else None),
                "narrower": narrower, "reserves_less": reserves_less,
                "dominates_here": bool(narrower and reserves_less),
                "extrapolated": bool(block["width_extrapolated"] or block["reserve_extrapolated"]),
                "alphas": block["alphas"],
            }
        report["positions"][position] = cell

    for name in loaded:
        cells = [c["alternatives"].get(name, {}) for c in report["positions"].values()]
        scored = [c for c in cells if c.get("present")]
        if scored and all(c["dominates_here"] for c in scored):
            dominating.append(name)
    report["dominates_everywhere"] = dominating

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "calibrator_comparison.json").write_text(json.dumps(report, indent=1),
                                                            encoding="utf-8", newline="\n")

    lines = ["# Calibrator substitution at matched coverage and matched risk", "",
             "Width is read where realized coverage reaches 0.90 and reservation where the realized",
             "exceedance rate reaches 0.10, both interpolated on each arm's own alpha sweep, so an arm",
             "cannot win by widening. `>1` means worse than the reported point-conformal arm.",
             "The reported arm at each position is the one the main table names: "
             + ", ".join(f"{TITLES.get(p, p)} `{a}`" for p, a in reported.items() if p in TITLES) + ".", ""]
    for position, reason in report["unscored"].items():
        lines.append(f"{TITLES[position]} is not scored: {reason}.")
    if report["unscored"]:
        lines.append("")
    lines += ["| position | alternative | width / ours | reserved / ours | narrower | reserves less | "
              "alphas | interpolated |", "|---|---|---:|---:|:--:|:--:|---|:--:|"]
    for position, cell in report["positions"].items():
        for name, alternative in cell["alternatives"].items():
            if not alternative.get("present"):
                lines.append(f"| {TITLES[position]} | {name} | n/a | n/a | | | none | |")
                continue
            width = f"{alternative['width_ratio']:.2f}" if alternative["width_ratio"] else "n/a"
            reserved = f"{alternative['reserved_ratio']:.2f}" if alternative["reserved_ratio"] else "n/a"
            lines.append(
                f"| {TITLES[position]} | {name} | {width} | {reserved} | "
                f"{'yes' if alternative['narrower'] else 'no'} | "
                f"{'yes' if alternative['reserves_less'] else 'no'} | "
                f"{'/'.join(alternative['alphas'])} | "
                f"{'yes' if alternative['extrapolated'] else 'no'} |")
    lines += ["", "## Verdict", ""]
    scored_titles = ", ".join(TITLES[p] for p in report["positions"])
    if dominating:
        lines.append("Alternatives at least as good on both axes at every scored position ("
                     + scored_titles + "): " + ", ".join(f"**{n}**" for n in dominating) + ".")
    else:
        lines.append("No alternative is at least as good on both width and reservation at every")
        lines.append(f"scored position ({scored_titles}), so the reported point-conformal choice")
        lines.append("is not dominated.")
    local = {name: [TITLES[p] for p, c in report["positions"].items()
                    if c["alternatives"].get(name, {}).get("dominates_here")]
             for name in loaded}
    local = {name: where for name, where in local.items() if where and name not in dominating}
    if local:
        lines.append("")
        lines.append("Alternatives that win on both axes at some positions and lose at others: "
                     + "; ".join(f"**{n}** at {', '.join(w)}" for n, w in local.items()) + ".")
    lines.append("")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
