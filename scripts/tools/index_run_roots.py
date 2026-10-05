from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_FILES = ("grid.json", "metrics.json", "comparisons.json", "run.json", "summary.md")
PREFIX_MEANING = {
    "cs": "call-start feature and method rounds",
    "ic": "in-call update rounds",
    "tu": "task-update rounds",
    "3p": "three-position joint rounds",
    "fp": "four-position rounds",
    "task-init": "task-initialization rounds",
    "rno": "reasoning-off collection",
    "msa": "SWE-bench rollouts collected with the public corpus scaffold",
    "asc": "unseen agent LLM adaptation (actor scale calibration) rounds",
    "swe-seeds": "SWE-bench task pack seed worktrees",
    "domain-seeds": "MMLU-Pro and LongBench seed workspaces",
}


def describe(directory: Path) -> dict:
    size = files = models = 0
    summaries: set[str] = set()
    config_id = ""
    for dirpath, dirnames, filenames in os.walk(directory):
        base = os.path.basename(dirpath)
        if base == "models":
            models += 1
            dirnames[:] = []
        for name in filenames:
            files += 1
            try:
                size += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                continue
            if name in SUMMARY_FILES:
                summaries.add(name)
            if name == "run.json" and not config_id:
                try:
                    config_id = str(json.loads(Path(dirpath, name).read_text(encoding="utf-8")).get("config_id") or "")
                except (OSError, ValueError):
                    config_id = ""
    return {
        "name": directory.name,
        "bytes": size,
        "files": files,
        "models": models,
        "summaries": sorted(summaries),
        "config_id": config_id,
    }


def prefix_of(name: str) -> str:
    for prefix in sorted(PREFIX_MEANING, key=len, reverse=True):
        if name == prefix or name.startswith(prefix + "-"):
            return prefix
    return ""


def run(root: Path, out: Path) -> None:
    entries = [describe(child) for child in sorted(root.iterdir()) if child.is_dir()]
    total = sum(entry["bytes"] for entry in entries)
    lines = [
        "# Experiment output roots",
        "",
        f"Index of `{root.as_posix()}`, written by `scripts/tools/index_run_roots.py`. "
        f"{len(entries)} directories, {total / 1e9:.1f} GB.",
        "",
        "Model bundles and prediction dumps are pruned once a summary exists; a run with "
        "no model directory still has its metrics.",
        "",
        "An empty `comparisons.json` is not damage. The file carries the paired bootstrap of "
        "every candidate that declares an `ablation`, so a config with none -- the main table "
        "among them -- writes `{}`. A zero-byte `predictions.jsonl` beside nothing else is a "
        "run that died at its first write, and those directories are deleted.",
        "",
        "| directory | GB | files | models | summaries | config id |",
        "|---|---|---|---|---|---|",
    ]
    for entry in sorted(entries, key=lambda item: -item["bytes"]):
        lines.append(
            f"| {entry['name']} | {entry['bytes'] / 1e9:.2f} | {entry['files']} | "
            f"{entry['models']} | {', '.join(entry['summaries']) or 'none'} | "
            f"{entry['config_id'] or ''} |"
        )
    grouped: dict[str, list[str]] = {}
    for entry in entries:
        grouped.setdefault(prefix_of(entry["name"]), []).append(entry["name"])
    lines += ["", "## Naming prefixes", "", "| prefix | meaning | count |", "|---|---|---|"]
    for prefix, meaning in PREFIX_MEANING.items():
        lines.append(f"| `{prefix}` | {meaning} | {len(grouped.get(prefix, []))} |")
    other = grouped.get("", [])
    if other:
        lines.append(f"| (none) | one-off roots | {len(other)} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{out}: {len(entries)} directories, {total / 1e9:.1f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(description='Write docs/run-roots.md, the index of every experiment output directory.'.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT / "workspace" / "runs")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "run-roots.md")
    arguments = parser.parse_args()
    run(arguments.root, arguments.out)


if __name__ == "__main__":
    main()
