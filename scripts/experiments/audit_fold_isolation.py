from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def scan(fit_dir: Path) -> dict:

    task_folds: dict[str, set[int]] = defaultdict(set)
    family_of: dict[str, set[str]] = defaultdict(set)
    repository_of: dict[str, str] = {}
    rows = 0
    with (fit_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            task = record.get("task_id")
            if not task:
                continue
            rows += 1
            if record.get("fold") is not None:
                task_folds[task].add(int(record["fold"]))
            if record.get("task_family_id"):
                family_of[task].add(record["task_family_id"])
            if record.get("repository"):
                repository_of[task] = record["repository"]
    return {"task_folds": task_folds, "family_of": family_of,
            "repository_of": repository_of, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grid", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict = {"grids": [], "verdict": {}}
    all_clean = True
    for grid in args.grid:
        splits = sorted(grid.glob("split-*/fit-*"))
        if not splits:
            print(f"no scored splits under {grid}", flush=True)
            continue
        block: dict = {"grid": grid.name, "splits": {}}
        assignments: dict[str, dict[str, int]] = {}
        for fit in splits:
            seed = fit.parents[0].name
            found = scan(fit)
            multi = {t: sorted(f) for t, f in found["task_folds"].items() if len(f) > 1}
            mixed = {t: sorted(f) for t, f in found["family_of"].items() if len(f) > 1}
            differs = {t: sorted(f)[0] for t, f in found["family_of"].items()
                       if sorted(f) and sorted(f)[0] != t}
            per_fold: dict[int, int] = defaultdict(int)
            for task, folds in found["task_folds"].items():
                for fold in folds:
                    per_fold[fold] += 1
            assignments[seed] = {t: sorted(f)[0] for t, f in found["task_folds"].items() if f}

            repo_folds: dict[str, set[int]] = defaultdict(set)
            for task, repository in found["repository_of"].items():
                repo_folds[repository] |= found["task_folds"].get(task, set())
            straddling = {r: sorted(f) for r, f in repo_folds.items() if len(f) > 1}
            tasks_in_straddling = sum(1 for t, r in found["repository_of"].items() if r in straddling)

            block["splits"][seed] = {
                "rows": found["rows"],
                "tasks": len(found["task_folds"]),
                "tasks_per_fold": dict(sorted(per_fold.items())),
                "tasks_in_more_than_one_fold": multi,
                "tasks_with_more_than_one_family": mixed,
                "tasks_whose_family_differs_from_the_task": differs,
                "repositories": len(repo_folds),
                "repositories_spanning_folds": len(straddling),
                "tasks_in_a_repository_that_spans_folds": tasks_in_straddling,
            }
            if multi or mixed:
                all_clean = False

        seeds = sorted(assignments)
        pairs = []
        for i, a in enumerate(seeds):
            for b in seeds[i + 1:]:
                shared = set(assignments[a]) & set(assignments[b])
                same = sum(1 for t in shared if assignments[a][t] == assignments[b][t])
                pairs.append({"seeds": [a, b], "shared_tasks": len(shared),
                              "same_fold": same,
                              "agreement": same / len(shared) if shared else None})
        block["seed_pairs"] = pairs
        report["grids"].append(block)

    report["verdict"]["no_task_scored_in_two_folds"] = all_clean

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "fold_isolation.json").write_text(json.dumps(report, indent=1),
                                                     encoding="utf-8", newline="\n")

    lines = ["# Fold isolation, checked against the scored rows", "",
             "Every number below is counted from `predictions.jsonl`, not from the split plan.", ""]
    for block in report["grids"]:
        lines += [f"## {block['grid']}", "",
                  "| split seed | rows | Tasks | Tasks per fold | in >1 fold | >1 family | "
                  "repositories | spanning folds | Tasks affected |", "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
        for seed, cell in block["splits"].items():
            spread = "/".join(str(v) for v in cell["tasks_per_fold"].values())
            lines.append(
                f"| {seed} | {cell['rows']:,} | {cell['tasks']} | {spread} | "
                f"{len(cell['tasks_in_more_than_one_fold'])} | {len(cell['tasks_with_more_than_one_family'])} | "
                f"{cell['repositories']} | {cell['repositories_spanning_folds']} | "
                f"{cell['tasks_in_a_repository_that_spans_folds']} |")
        lines.append("")
        if block["seed_pairs"]:
            lines += ["| seed pair | shared Tasks | same fold | agreement |", "|---|---:|---:|---:|"]
            for pair in block["seed_pairs"]:
                agreement = f"{pair['agreement']:.3f}" if pair["agreement"] is not None else "n/a"
                lines.append(f"| {pair['seeds'][0]} vs {pair['seeds'][1]} | {pair['shared_tasks']} | "
                             f"{pair['same_fold']} | {agreement} |")
            lines.append("")
    lines += ["## Verdict", "",
              f"- No Task scored in more than one fold: **{all_clean}**.",
              "- A Task's family id equals its own id in this corpus, so family isolation and Task",
              "  isolation are the same constraint here. The registry is built to hold a coarser",
              "  family when a corpus supplies one.",
              "- Repositories do span folds. Family isolation does not control for that by design;",
              "  the leave-one-repository-out experiment is the evidence for repository transfer,",
              "  and a random-fold agreement near 0.2 is what five folds should give.", ""]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
