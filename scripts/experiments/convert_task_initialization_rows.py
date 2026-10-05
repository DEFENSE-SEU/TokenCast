from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grid", type=Path)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", default="")
    args = parser.parse_args()
    wanted = set(args.candidates.split(",")) if args.candidates else None
    transfer: dict[int, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    with args.transfer.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if wanted is not None and record["candidate_id"] not in wanted:
                continue
            transfer[int(record["seed"])][record["candidate_id"]][record["task_id"]] = record
    written = defaultdict(int)
    for fit_dir in sorted(args.grid.glob("split-*/fit-*")):
        seed = int(fit_dir.parent.name.split("-")[1])
        if seed not in transfer:
            continue
        out_dir = args.output / fit_dir.parent.name / fit_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        seen = set()
        with (fit_dir / "predictions.jsonl").open(encoding="utf-8") as handle, (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as sink:
            for line in handle:
                record = json.loads(line)
                if not record["experiment_id"].startswith("task_initialization--") or record["point_id"] in seen:
                    continue
                seen.add(record["point_id"])
                for candidate, by_task in transfer[seed].items():
                    source = by_task.get(record["task_id"])
                    if source is None:
                        continue
                    sink.write(json.dumps({**{k: record[k] for k in ("point_id", "experiment_id", "run_id", "task_id", "fold", "target_value", "sample_weight")}, "candidate_id": candidate, "prediction": float(source["point"]), "lower": float(source["lower"]), "upper": float(source["upper"]), "label_status": "observed", "latency_ms": 0.0}) + "\n")
                    written[candidate] += 1
    print({k: v for k, v in written.items()})


if __name__ == "__main__":
    main()
