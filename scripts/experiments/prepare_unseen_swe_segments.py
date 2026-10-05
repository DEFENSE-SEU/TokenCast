from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from token_prediction.case_study.qwen38_adapter import family_id
from token_prediction.case_study.segments import load_run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--exclude-provenance", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError("output must be empty")
    excluded = json.loads(a.exclude_provenance.read_text(encoding="utf-8"))["task_family_partition"]
    excluded = {family_id(f) for f in excluded}
    selected = {}
    for directory in sorted(a.runs_root.iterdir()):
        if not (directory / "state.json").exists():
            continue
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        if state.get("capture_status") not in {"complete", "validated"} or not state.get(
            "supervision_eligible"
        ):
            continue
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        family = family_id(run["task"]["task_family_id"])
        if family not in excluded and family not in selected:
            selected[family] = str(directory.resolve())
    a.output.mkdir(parents=True)
    prov = {
        "task_family_partition": {family: "test" for family in selected},
        "sources": selected,
        "paid_api_calls": 0,
        "test_evaluated": False,
        "excluded_provenance_sha256": hashlib.sha256(a.exclude_provenance.read_bytes()).hexdigest(),
        "selection": "first complete supervision-eligible run per unused family in sorted directory order",
    }
    (a.output / "protocol.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    batches, rejected = [], []
    for family, directory in selected.items():
        try:
            batch = load_run(directory, request_surface_features=True)
            call = np.isin(batch["position"], ["call_start", "in_call_update"])
            if not batch["request_surface_complete"][call].all():
                raise ValueError("incomplete request surface")
        except (ValueError, OSError) as e:
            rejected.append({"family": family, "reason": str(e)})
            continue
        batch["task_family_id"] = np.full(len(batch["n"]), family)
        run = json.loads((Path(directory) / "run.json").read_text(encoding="utf-8"))
        batch["model_id"] = np.full(len(batch["n"]), run["condition"]["actor_model_id"])
        batch["harness"] = np.full(len(batch["n"]), "dsh")
        batch["benchmark"] = np.full(len(batch["n"]), "swebench")
        batches.append(batch)
        print(json.dumps({"accepted_families": len(batches)}), flush=True)
    if not batches:
        raise ValueError("no unused complete tasks")
    data = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
    np.savez_compressed(a.output / "test.npz", **data)
    prov["rejected"] = rejected
    prov["accepted_families"] = len(batches)
    prov["test_sha256"] = hashlib.sha256((a.output / "test.npz").read_bytes()).hexdigest()
    (a.output / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(json.dumps({"accepted_families": len(batches), "rejected": len(rejected)}), flush=True)


if __name__ == "__main__":
    main()
