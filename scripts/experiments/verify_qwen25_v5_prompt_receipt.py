from __future__ import annotations

import hashlib
import json
from pathlib import Path

from token_prediction.collection.dsh.jsonl import physical_path, read_jsonl_with_tail


ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "workspace/experiments/local_qwen25_v5_exposed_preflight_20260923"
COLLECTION = ROOT / "workspace/collections/qwen25-v5-exposed-preflight-20260923"
POLICY = "swebench-source-patch-submit-v5"
PROMPT_MARKERS = (
    f"DSH tool guidance policy: {POLICY}.",
    "There is no `ls` tool.",
    "do not retry that path or guess a nearby one",
    "Search from the checkout root with `glob`",
)


def tool_name(item: dict) -> str | None:
    direct = item.get("name")
    if isinstance(direct, str):
        return direct
    nested = item.get("function")
    if isinstance(nested, dict) and isinstance(nested.get("name"), str):
        return nested["name"]
    return None


def main() -> None:
    lock = json.loads((DESIGN / "freeze.json").read_text(encoding="utf-8"))
    manifest_path = DESIGN / "manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != lock["v5_manifest_sha256"]:
        raise RuntimeError("v5 manifest differs from its development freeze")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest["actor"]["system_prompt_policy_id"] != POLICY
        or manifest["observer"]["request_content_ledger"] is not True
        or manifest["model_conditions"][0]["configured_provider"] != "local-qwen"
    ):
        raise RuntimeError("v5 manifest policy, route, or request receipt changed")
    run_id = lock["one_pilot_run_id"]
    collection = json.loads((COLLECTION / "collection.json").read_text(encoding="utf-8"))
    run_dir = COLLECTION / collection["run_directories"][run_id]
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    if not state["launches"]:
        raise RuntimeError("v5 exposed pilot has no actor launch yet")
    launch = state["launches"][0]
    session_id = launch["session_id"]
    capture = run_dir / launch["capture_path"] / "request-content.jsonl"
    staged = COLLECTION / "batch/observer/request-content" / f"{session_id}.jsonl"
    paths = [path for path in (capture, staged) if physical_path(path).is_file()]
    if not paths:
        raise RuntimeError("the first Request content receipt is not durable yet")
    if len(paths) == 2 and physical_path(paths[0]).read_bytes() != physical_path(paths[1]).read_bytes():
        raise RuntimeError("staged and archived Request content receipts conflict")
    source = physical_path(paths[0])
    rows, _tail = read_jsonl_with_tail(source, allow_incomplete=True)
    records = [row for row in rows if row.get("record_type") == "request/content"]
    if not records:
        raise RuntimeError("Request content receipt has no complete first record")
    first = records[0]
    request = first.get("request") or {}
    system = request.get("system")
    if (
        first.get("session_id") != session_id
        or first.get("attempt_ordinal") != 1
        or first.get("status") != "complete"
        or first.get("missing_fields")
        or request.get("provider") != "local-qwen"
        or request.get("model") != "Qwen/Qwen2.5-7B-Instruct"
        or not isinstance(system, str)
        or any(marker not in system for marker in PROMPT_MARKERS)
    ):
        raise RuntimeError("the first serialized Request does not prove v5 guidance")
    tools = request.get("tools")
    if not isinstance(tools, list):
        raise RuntimeError("the first Request has no explicit tool roster")
    names = {tool_name(item) for item in tools if isinstance(item, dict)}


    if names != {"edit", "pwsh", "read", "write"}:
        raise RuntimeError(f"the first Request tool roster differs: {sorted(names, key=str)}")
    receipt = {
        "schema_version": 1,
        "status": "v5_first_request_verified",
        "run_id": run_id,
        "session_id": session_id,
        "system_prompt_policy_id": POLICY,
        "first_request_hash": first["request_hash"],
        "system_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
        "tool_names": sorted(names),
        "request_content_path": str(source),
        "request_content_sha256_at_verification": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    out = DESIGN / "v5_first_request_receipt.json"
    encoded = (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if out.exists():
        prior = json.loads(out.read_text(encoding="utf-8"))
        if any(prior[key] != receipt[key] for key in (
            "status", "run_id", "session_id", "system_prompt_policy_id",
            "first_request_hash", "system_sha256", "tool_names",
        )):
            raise RuntimeError("the saved first-Request receipt conflicts with current evidence")
    else:
        with out.open("xb") as stream:
            stream.write(encoded)
    print(json.dumps({"status": receipt["status"], "run_id": run_id,
                      "request_hash": receipt["first_request_hash"],
                      "tool_names": receipt["tool_names"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
