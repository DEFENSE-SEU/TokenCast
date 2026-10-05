from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import requests
from huggingface_hub import HfApi


REPO = "JetBrains-Research/EnvBench-trajectories"
REVISION = "4ffd02bed0f39fd87bffe3be7edf2440aba16cd1"
RUN = "python/bash_agent-4o"
PROBE = "skrub-data__skrub@0d69d97129aa638167e1e721686acb066daed007"
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "workspace/experiments/envbench_python_second_source_20260923/source_feasibility.json"
JVM_LOCK = ROOT / "workspace/experiments/envbench_budget_feasibility_20260923/jvm_bash_agent_4o_cohort.lock.json"


def git_blob_oid(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def main() -> None:
    api = HfApi()
    listing = list(api.list_repo_tree(REPO, path_in_repo=RUN, recursive=False, revision=REVISION, repo_type="dataset"))
    result = next(row for row in listing if row.path == f"{RUN}/results.jsonl")
    trajectories = list(
        api.list_repo_tree(
            REPO, path_in_repo=f"{RUN}/trajectories", recursive=True, revision=REVISION, repo_type="dataset"
        )
    )
    records = {row.path.rsplit("/", 1)[-1][:-6]: row for row in trajectories}
    assert len(records) == len(trajectories) == 328
    assert len({name.split("@")[0] for name in records}) == 328
    assert PROBE in records
    assert result.lfs is not None

    jvm_lock = json.loads(JVM_LOCK.read_text(encoding="utf-8"))
    jvm_ids = set(jvm_lock["trajectory_metadata"])
    jvm_repos = {name.split("@")[0] for name in jvm_ids}
    assert not (set(records) & jvm_ids)
    assert not ({name.split("@")[0] for name in records} & jvm_repos)

    result_url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{RUN}/results.jsonl"


    with requests.get(result_url, headers={"Range": "bytes=0-266260"}, stream=True, timeout=60) as response:
        response.raise_for_status()
        assert response.status_code == 206
        result_first_line = response.raw.readline(266_261)
    assert result_first_line.endswith(b"\n")
    first = json.loads(result_first_line)
    assert f"{first['repo_name'].replace('/', '__')}@{first['commit_sha']}" == PROBE

    probe_row = records[PROBE]
    probe_url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{probe_row.path}"
    probe_bytes = requests.get(probe_url, timeout=60).content
    assert len(probe_bytes) == probe_row.size
    assert git_blob_oid(probe_bytes) == probe_row.blob_id
    events = [json.loads(line) for line in probe_bytes.splitlines()]
    messages = [m for event in events for m in event.get("messages", [])]
    ai = [m for m in messages if m.get("message_type") == "ai"]
    tools = [m for m in messages if m.get("message_type") == "tool"]
    usage = [m.get("message_content", {}).get("usage_metadata") for m in ai]
    usage_valid = [
        isinstance(u, dict)
        and all(isinstance(u.get(k), int) and u[k] >= 0 for k in ("input_tokens", "output_tokens", "total_tokens"))
        and u["input_tokens"] + u["output_tokens"] == u["total_tokens"]
        for u in usage
    ]
    assert all(usage_valid)

    receipt = {
        "state": "source_feasibility_only_no_test_read",
        "dataset": REPO,
        "revision": REVISION,
        "run": RUN,
        "trajectory_count": len(records),
        "candidate_after_probe_exclusion": len(records) - 1,
        "trajectory_bytes_total": sum(row.size for row in trajectories),
        "unique_repositories": len({name.split("@")[0] for name in records}),
        "overlap_with_jvm_exact_ids": len(set(records) & jvm_ids),
        "overlap_with_jvm_repository_names": len({name.split("@")[0] for name in records} & jvm_repos),
        "results_metadata": {
            "path": result.path,
            "size": result.size,
            "blob_id": result.blob_id,
            "lfs_sha256": result.lfs.sha256,
        },
        "excluded_probe": {
            "selection": "first physical results.jsonl line, then matching trajectory",
            "task_id": PROBE,
            "first_result_line_bytes": len(result_first_line),
            "first_result_fields": sorted(first),
            "first_result_field_types": {key: type(value).__name__ for key, value in first.items()},
            "pyright_fields": sorted(first.get("pyright", {})),
            "trajectory_path": probe_row.path,
            "trajectory_size": probe_row.size,
            "trajectory_git_blob_oid": probe_row.blob_id,
            "trajectory_sha256": hashlib.sha256(probe_bytes).hexdigest(),
            "event_count": len(events),
            "event_nodes": dict(collections.Counter(event.get("node") for event in events)),
            "ai_messages": len(ai),
            "tool_messages": len(tools),
            "valid_ai_usage": sum(usage_valid),
            "tool_messages_with_string_content": sum(
                isinstance(m.get("message_content", {}).get("content"), str) for m in tools
            ),
            "finish_reasons": dict(collections.Counter(m.get("response_metadata", {}).get("finish_reason") for m in ai)),
            "model_names": dict(collections.Counter(m.get("response_metadata", {}).get("model_name") for m in ai)),
            "ai_tool_call_counts": [len(m.get("message_content", {}).get("tool_calls", [])) for m in ai],
        },
        "trajectory_metadata": {
            name: {"path": row.path, "size": row.size, "blob_id": row.blob_id}
            for name, row in sorted(records.items())
        },
        "test_trajectory_or_result_opened": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)
    print(hashlib.sha256(OUTPUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
