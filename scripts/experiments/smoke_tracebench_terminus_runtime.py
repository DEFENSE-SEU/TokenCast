import json
from collections import Counter

from qualify_tracebench_terminus_development import WORK, sha256
from tracebench_terminus_point_runtime import episodes_from_archive, pre_gate, request_view, response_view


def main() -> None:
    audit_path = WORK / "development_quality_audit_v2.json"
    audit = json.loads(audit_path.read_text())
    index = json.loads((WORK / "terminus_artifact_index.metadata.json").read_text())
    lock = json.loads((WORK / "terminus_point_protocol.predev.lock.json").read_text())
    development_ids = set(lock["fit_task_ids"]) | set(lock["calibration_task_ids"])
    rows = [row for row in index["artifacts"] if row["task_name"] in development_ids]
    expected = {row["artifact_path"]: row for row in audit["trajectories"]}
    if len(rows) != 356 or set(expected) != {row["artifact_path"] for row in rows}:
        raise RuntimeError("Old-identity audit roster mismatch")
    totals = Counter()
    for row in rows:
        path = WORK / "development_archives" / row["artifact_path"].split("/")[-1]
        episodes = episodes_from_archive(row, path)
        previous = None
        seen_response_ids = set()
        seen_call_ids = set()
        pre_count = 0
        score_count = 0
        for episode_id, debug_bytes, prompt in episodes:
            current = {"request": request_view(debug_bytes, prompt, episode_id, row["model"])}
            if previous is not None:
                failures = pre_gate(previous, current, seen_response_ids, seen_call_ids)
                if not failures:
                    pre_count += 1
                prior = previous["response"]
                if prior["response_id"]:
                    seen_response_ids.add(prior["response_id"])
                if prior["litellm_call_id"]:
                    seen_call_ids.add(prior["litellm_call_id"])
            current["response"] = response_view(debug_bytes)
            if previous is not None and not failures:
                response = current["response"]
                if (response["usage_valid"] and response["response_valid"]
                    and not (response["response_id"] and response["response_id"] in seen_response_ids)
                    and not (response["litellm_call_id"] and response["litellm_call_id"] in seen_call_ids)):
                    score_count += 1
            previous = current
        prior_audit = expected[row["artifact_path"]]
        if pre_count != prior_audit["prediction_pre_gate_count"] or score_count != prior_audit["eligible_pair_count"]:
            raise RuntimeError(f"Gate mismatch {row['traj_id']}: runtime {pre_count}/{score_count}, development {prior_audit['prediction_pre_gate_count']}/{prior_audit['eligible_pair_count']}")
        totals["pre_gate"] += pre_count
        totals["score_gate"] += score_count
        totals["archives"] += 1
    report = {
        "status": "old_identity_runtime_gate_match",
        "development_audit_sha256": sha256(audit_path),
        "archives": totals["archives"],
        "pre_gate_points": totals["pre_gate"],
        "score_gate_points": totals["score_gate"],
        "test_content_read": False,
    }
    path = WORK / "runtime_development_smoke_v5.json"
    if path.exists():
        raise RuntimeError("Smoke receipt already exists")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"runtime gate match on {totals['archives']} old archives, {totals['pre_gate']} pre-gate calls; receipt SHA256 {sha256(path)}")


if __name__ == "__main__":
    main()
