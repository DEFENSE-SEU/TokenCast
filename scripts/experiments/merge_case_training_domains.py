from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from token_prediction.case_study.qwen38_adapter import family_id
from token_prediction.case_study.segments import request_group_id, with_history_cost_observations
from token_prediction.collection.readers.dsh_session import DshSessionReader


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--historical", type=Path, required=True)
    p.add_argument("--qwen", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--history-cost-features", action="store_true",
                   help="Add settled-call usage to causal Task Update observations (history v3).")
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise ValueError("output must be empty")
    old_prov = json.loads((a.historical / "provenance.json").read_text(encoding="utf-8"))
    new_prov = json.loads((a.qwen / "provenance.json").read_text(encoding="utf-8"))
    old = dict(np.load(a.historical / "segments.npz"))
    new = dict(np.load(a.qwen / "training_segments.npz"))
    partition = {family_id(k): v for k, v in old_prov["task_family_partition"].items()}
    for family, split in new_prov["task_family_partition"].items():
        family = family_id(family)
        if family in partition and partition[family] != split:
            raise ValueError("existing task partition changed")
        partition[family] = split
    conditions, task_sizes = {}, {}
    for source in old_prov["sources"]:
        run = json.loads((Path(source) / "run.json").read_text(encoding="utf-8"))
        conditions[family_id(run["task"]["task_family_id"])] = run["condition"]["actor_model_id"]
        root = Path(source)
        state = json.loads((root / "state.json").read_text(encoding="utf-8"))
        trajectory = DshSessionReader().read(root / state["capture_path"])
        text = str(trajectory.events[0].payload.get("task_text", ""))
        task_sizes[family_id(run["task"]["task_family_id"])] = len(text.encode("utf-8"))
    for batch in (old, new):
        batch["task_family_id"] = np.array([family_id(x) for x in batch["task_family_id"]])
        batch["completed_calls"] = batch["start_call"].copy()
        assert all(partition[family] != "test" for family in batch["task_family_id"])
    old["model_id"] = np.array([conditions[f] for f in old["task_family_id"]])
    old["task_text_bytes"] = np.array([task_sizes[f] for f in old["task_family_id"]])
    assert "task_text_bytes" in new, "Qwen task-size enrichment required"
    old["harness"] = np.full(len(old["n"]), "dsh")
    old["benchmark"] = np.full(len(old["n"]), "swebench")
    for batch in (old, new):
        batch["request_group"] = np.array(
            [request_group_id(str(model)) for model in batch["model_id"]]
        )
    keys = sorted(set(old) & set(new))
    combined = {key: np.concatenate([old[key], new[key]]) for key in keys}
    if len(set(combined["point_id"])) != len(set(old["point_id"])) + len(set(new["point_id"])):
        raise ValueError("source point identifiers overlap")
    if a.history_cost_features:
        combined = with_history_cost_observations(combined)
    a.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output / "segments.npz", **combined)
    counts = {}
    for split in ("train", "validation", "calibration"):
        mask = np.array([partition[f] == split for f in combined["task_family_id"]])
        counts[split] = {
            "rows": int(mask.sum()),
            "families": len(set(combined["task_family_id"][mask])),
            "runs": len(set(combined["run_id"][mask])),
        }
    prov = {
        "feature_schema": ("causal_request_surface_history_v3" if a.history_cost_features
                           else "causal_request_surface_v2"),
        "paid_api_calls": 0,
        "task_family_partition": partition,
        "counts": counts,
        "sources": old_prov["sources"],
        "additional_source_provenance": str((a.qwen / "provenance.json").resolve()),
        "test_data_read": False,
        "selection": "prelocked family splits; all preselected development runs",
        "datasets_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (a.historical / "segments.npz", a.qwen / "training_segments.npz")
        },
    }
    (a.output / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
