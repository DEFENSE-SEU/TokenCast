from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "workspace/experiments/envbench_budget_feasibility_20260923/jvm_bash_agent_4o_cohort.lock.json"
REPO = "JetBrains-Research/EnvBench-trajectories"
REVISION = "4ffd02bed0f39fd87bffe3be7edf2440aba16cd1"
RUN = "jvm/bash_agent-4o"
PROBE_ID = "celzero__rethink-app@6fc4c28d02cb3b84066b7e0dc432096c4a494ad6"
NAMESPACE = b"envbench-jvm-bash-agent-4o-v1\0"


def tree(path: str) -> list[dict]:
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{path}"
    for attempt in range(5):
        try:
            response = requests.get(url, params={"limit": 1000}, timeout=45)
            response.raise_for_status()
            if response.links.get("next"):
                raise ValueError("Unpaged source tree")
            return response.json()
        except requests.RequestException:
            if attempt == 4:
                raise
    raise AssertionError("unreachable")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    trajectories = tree(f"{RUN}/trajectories")
    if len(trajectories) != 665 or any(
        entry["type"] != "file" or not entry["path"].endswith(".jsonl")
        for entry in trajectories
    ):
        raise ValueError("Source trajectory roster differs from metadata survey")
    roster = {
        entry["path"].rsplit("/", 1)[-1][:-6]: {
            "path": entry["path"], "bytes": entry["size"], "git_blob_oid": entry["oid"]
        }
        for entry in trajectories
    }
    if len(roster) != 665 or len({task_id.split("@")[0] for task_id in roster}) != 665:
        raise ValueError("Task or repository identities collide")
    if PROBE_ID not in roster:
        raise ValueError("Schema probe missing")
    result = next(
        entry for entry in tree(RUN) if entry["path"] == f"{RUN}/results.jsonl"
    )
    if result.get("lfs", {}).get("oid") != "d29403a2762da63f1e18d26e579c99f2d715fc575e43744d7c5251856f8ec04d":
        raise ValueError("Evaluation result object changed")
    ids = sorted(
        (task_id for task_id in roster if task_id != PROBE_ID),
        key=lambda task_id: (sha256(NAMESPACE + task_id.encode()).hexdigest(), task_id),
    )
    if len(ids) != 664:
        raise ValueError("Unexpected cohort size")
    payload = {
        "state": "task_identity_split_frozen_from_metadata_before_new_test_outcomes",
        "repo": REPO,
        "revision": REVISION,
        "run": RUN,
        "source_trajectory_count": 665,
        "schema_probe_excluded_id": PROBE_ID,
        "cohort_identity": "one repo@commit trajectory from one JVM Bash Agent GPT-4o run",
        "hash_order": "SHA256(UTF8('envbench-jvm-bash-agent-4o-v1\\0') || UTF8(task_id)), ascending",
        "fit_ids": ids[:400],
        "calibration_ids": ids[400:532],
        "untouched_test_ids": ids[532:],
        "trajectory_metadata": {task_id: roster[task_id] for task_id in sorted(roster)},
        "result_metadata": {
            "path": result["path"],
            "bytes": result["size"],
            "git_pointer_oid": result["oid"],
            "lfs_sha256": result["lfs"]["oid"],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(json.dumps({
        "lock": str(OUT),
        "sha256": sha256(OUT.read_bytes()).hexdigest(),
        "fit": len(payload["fit_ids"]),
        "calibration": len(payload["calibration_ids"]),
        "untouched_test": len(payload["untouched_test_ids"]),
    }, indent=2))


if __name__ == "__main__":
    main()
