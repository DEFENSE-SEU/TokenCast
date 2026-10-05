from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import tiktoken

from .develop_envbench_jvm_budget_models import BASE, digest, read_jsonl, write_json_exclusive


OUT = BASE / "development_token_transport_prefix_mutation.json"
CASES = (
    ("fit", "8vim__8vim@fe4452e1683a9ec0faec78cff164e3313fcbef8b", 1),
    ("calibration", "killbill__killbill-commons@9e49ff3b58b55f687e4e7180b889d536e1dbb00d", 27),
)


def prefix_prediction(events: list[dict], target: tuple[int, int], encoding) -> dict:
    prior = None
    tools = 0
    tool_tokens = 0
    for event_index, event in enumerate(events):
        for message_index, message in enumerate(event.get("messages", [])):
            if (event_index, message_index) == target:
                if message.get("message_type") != "ai" or prior is None:
                    raise ValueError("Target is not a later AI Call Start")
                return {"prediction": prior[0] + prior[1] + tool_tokens + 9,
                        "prior_usage": list(prior), "tool_messages": tools,
                        "tool_content_tokens": tool_tokens}
            kind = message.get("message_type")
            if kind == "tool":
                content = message["message_content"]["content"]
                if not isinstance(content, str):
                    raise ValueError("Tool content is not text")
                tools += 1
                tool_tokens += len(encoding.encode(content))
            elif kind == "ai":
                usage = message["message_content"].get("usage_metadata")
                if usage is None:
                    raise ValueError("Unexpected pre-target missing usage")
                prior = (usage["input_tokens"], usage["output_tokens"])
                tools = 0
                tool_tokens = 0
    raise ValueError("Target Call Start not found")


def main() -> None:
    if OUT.exists():
        raise FileExistsError("Prefix mutation report already exists")
    encoding = tiktoken.encoding_for_model("gpt-4o-2024-11-20")
    checked = []
    for group, task_id, call_index in CASES:
        task_rows = read_jsonl(BASE / f"development_{group}_tasks_v2.jsonl")
        task = next(row for row in task_rows if row["task_id"] == task_id)
        path = BASE / f"raw_{group}" / f"{task_id}.jsonl"
        raw = path.read_bytes()
        if digest(path) != task["raw_sha256"]:
            raise ValueError("Development source SHA drift")
        events = [json.loads(line) for line in raw.splitlines() if line.strip()]
        real_ai = [(ei, mi) for ei, event in enumerate(events)
                   for mi, message in enumerate(event.get("messages", []))
                   if message.get("message_type") == "ai"
                   and message.get("message_content", {}).get("usage_metadata") is not None]
        target = real_ai[call_index]
        original = prefix_prediction(events, target, encoding)
        mutated = deepcopy(events)
        current = mutated[target[0]]["messages"][target[1]]
        current_content = current["message_content"]
        current_content["content"] = "MUTATED-CURRENT-OUTPUT" * 20_000
        current_content["tool_calls"] = [{"name": "mutated_current_call", "args": {"large": "x" * 2000}}]
        current_content["usage_metadata"] = {
            "input_tokens": 987_654_321, "output_tokens": 123_456_789,
            "total_tokens": 1_111_111_110,
        }
        current["response_metadata"] = {"model_name": "mutated", "token_usage": {"prompt_tokens": 987_654_321}}
        if prefix_prediction(mutated, target, encoding) != original:
            raise ValueError("Current AI output or usage affected pre-call prediction")
        current_content["usage_metadata"] = None
        if prefix_prediction(mutated, target, encoding) != original:
            raise ValueError("Current AI usage availability affected pre-call prediction")
        mutated[target[0]]["messages"].append({"message_type": "tool", "message_content": {
            "content": "FUTURE-TOOL-OUTPUT" * 5000}})
        if prefix_prediction(mutated, target, encoding) != original:
            raise ValueError("Future tool message affected pre-call prediction")
        checked.append({"group": group, "task_id": task_id, "call_index": call_index,
                        "target_source_position": list(target), **original,
                        "current_usage_output_toolcalls_mutated": True,
                        "current_usage_removed": True,
                        "future_tool_appended": True,
                        "prediction_unchanged": True})
    report = {
        "status": "development_only_prefix_mutation_pass",
        "script_sha256": digest(Path(__file__)),
        "test_trajectory_access": False,
        "run_level_results_access": False,
        "cases": checked,
        "limit": "Fixture current-message mutation verifies the parser's pre-call data boundary; it does not establish a deployed pre-dispatch hook.",
    }
    write_json_exclusive(OUT, report)
    print(json.dumps({"report_sha256": digest(OUT), "cases": checked}, indent=2))


if __name__ == "__main__":
    main()
