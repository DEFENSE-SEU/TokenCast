from __future__ import annotations

import hashlib
import json
from pathlib import Path

from token_prediction.collection.dsh.manifest import (
    condition_roster_digest,
    parse_dsh_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "workspace/experiments/local_qwen25_paired_roster_20260923/preflight_manifest.json"
OUT = ROOT / "workspace/experiments/local_qwen25_v6_exposed_preflight_20260923"
MANIFEST_ID = "swebench-local-qwen25-v6-exposed-preflight-20260923"
CONDITION_ID = "swebench-qwen2p5-7b-nf4-off-guidance-v6"
POLICY_ID = "swebench-source-patch-submit-v6"


def main() -> None:
    base_bytes = BASE.read_bytes()
    manifest = json.loads(base_bytes)
    task_ids = [task["task_id"] for task in manifest["tasks"]]
    if task_ids != [
        "swebench-django-django-10097",
        "swebench-django-django-10973",
        "swebench-astropy-astropy-14598",
    ]:
        raise RuntimeError("v6 development pilot must use the three exposed task families")
    if len(manifest["model_conditions"]) != 1 or (
        manifest["model_conditions"][0]["configured_provider"] != "local-qwen"
        or manifest["model_conditions"][0]["credential_source"] != "none"
    ):
        raise RuntimeError("v6 development pilot requires one credential-free local route")
    manifest["manifest_id"] = MANIFEST_ID
    manifest["protocol_id"] = "swebench-local-qwen25-tool-guidance-v6-preflight"
    manifest["actor"]["system_prompt_policy_id"] = POLICY_ID
    manifest["model_conditions"][0]["condition_id"] = CONDITION_ID
    manifest["schedule"]["condition_order"] = [CONDITION_ID]
    manifest["observer"]["request_content_ledger"] = True
    manifest["launch_blockers"] = [
        "single-exposed-task-preflight-only-before-any-new-family-collection",
        "verify-first-request-v6-prompt-and-initial-pwsh-tool-before-broad-run",
    ]
    manifest["collection_authorized"] = False
    parsed = parse_dsh_manifest(manifest)
    manifest["condition_digest"] = condition_roster_digest(parsed.model_conditions)
    parsed = parse_dsh_manifest(manifest)
    run_ids = [run.run_id for run in parsed.scheduled_runs()]
    if len(run_ids) != 3 or any(
        run.condition.configured_provider != "local-qwen"
        or run.system_prompt_policy_id != POLICY_ID
        for run in parsed.scheduled_runs()
    ):
        raise RuntimeError("v6 development matrix has a route or policy mismatch")
    raw = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    metadata = {
        "schema_version": 1,
        "status": "frozen_exposed_development_no_actor_run",
        "base_manifest_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "v6_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_id": MANIFEST_ID,
        "condition_id": CONDITION_ID,
        "system_prompt_policy_id": POLICY_ID,
        "request_content_ledger": True,
        "task_ids": task_ids,
        "run_ids": run_ids,
        "one_pilot_run_id": run_ids[0],
        "local_model_api_fee_cny": 0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for path, content in (
        (OUT / "manifest.json", raw),
        (OUT / "freeze.json", (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8")),
    ):
        if path.exists():
            if path.read_bytes() != content:
                raise RuntimeError(f"refusing to overwrite changed freeze: {path}")
        else:
            with path.open("xb") as stream:
                stream.write(content)
    print(json.dumps({"status": "frozen", "manifest": str(OUT / "manifest.json"),
                      "one_pilot_run_id": run_ids[0]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
