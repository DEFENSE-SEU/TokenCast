from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import statistics

import numpy as np
import tiktoken

from develop_envbench_additive_input import replay
from develop_envbench_jvm_budget_models import (
    BASE, CAPS_FILE, CAPS_SHA, FEATURES, LOCK, LOCK_SHA, V2, V2_SHA,
    development_data, digest,
)


OUT = BASE / "development_token_transport_probe.json"
ENCODING_MODEL = "gpt-4o"
IDX = {name: index for index, name in enumerate(FEATURES)}


def preceding_tool_content(raw: bytes, encoding) -> dict[int, tuple[int, int, int]]:
    observations = {}
    ai_index = 0
    tool_tokens = tool_count = payload_bytes = 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        for message in event.get("messages", []):
            if not isinstance(message, dict):
                raise ValueError("Nonobject source message")
            kind = message.get("message_type")
            if kind == "tool":
                body = message.get("message_content")
                if not isinstance(body, dict) or not isinstance(body.get("content"), str):
                    raise ValueError("Unrecognized tool content")
                tool_tokens += len(encoding.encode(body["content"]))
                tool_count += 1
                payload_bytes += len(json.dumps(body, ensure_ascii=False, sort_keys=True,
                                                separators=(",", ":")).encode("utf-8"))
            elif kind == "ai":
                observations[ai_index] = (tool_tokens, tool_count, payload_bytes)
                ai_index += 1
                tool_tokens = tool_count = payload_bytes = 0
    return observations


def rows_for_group(group: str, points: list[dict], tasks: dict[str, dict], encoding) -> list[dict]:
    by_id: dict[str, list[dict]] = defaultdict(list)
    for point in points:
        by_id[point["task_id"]].append(point)
    rows = []
    for task_id in sorted(by_id):
        path = BASE / f"raw_{group}" / f"{task_id}.jsonl"
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != tasks[task_id]["raw_sha256"]:
            raise ValueError("Development source SHA drift")
        prior = preceding_tool_content(raw, encoding)
        for point in by_id[task_id]:
            token_count, tool_count, payload_bytes = prior[point["call_index"]]
            feature = point["features"]
            if (tool_count != feature[IDX["tool_messages_received"]] or
                    payload_bytes != feature[IDX["tool_feedback_payload_bytes"]] or
                    point["previous_input"] != feature[IDX["previous_input"]]):
                raise ValueError("Tool ordering or prefix feature mismatch")
            baseline = (feature[IDX["previous_input"]] +
                        feature[IDX["previous_output"]] + token_count)
            rows.append({"task_id": task_id, "call_index": point["call_index"],
                         "tool_messages": tool_count, "tool_content_tokens": token_count,
                         "previous_input": point["previous_input"],
                         "previous_output": feature[IDX["previous_output"]],
                         "current_input_label": point["next_input_tokens"],
                         "framing_residual": point["next_input_tokens"] - baseline,
                         "baseline": baseline})
    if len(rows) != len(points):
        raise ValueError("Development point coverage failure")
    return rows


def summary(rows: list[dict], framing: int) -> dict:
    errors = np.asarray([abs(row["baseline"] + framing - row["current_input_label"])
                         for row in rows], dtype=float)
    labels = np.asarray([row["current_input_label"] for row in rows], dtype=float)
    return {"checkpoints": len(rows), "single_tool": sum(row["tool_messages"] == 1 for row in rows),
            "multiple_tool": sum(row["tool_messages"] > 1 for row in rows),
            "point_mae": float(errors.mean()), "wape": float(errors.sum() / labels.sum()),
            "median_absolute_error": float(np.median(errors)),
            "p95_absolute_error": float(np.quantile(errors, .95, method="nearest")),
            "max_absolute_error": float(errors.max()),
            "within_two_tokens": int((errors <= 2).sum()),
            "single_tool_residual_counts": {
                str(key): value for key, value in sorted(Counter(
                    int(row["framing_residual"]) for row in rows
                    if row["tool_messages"] == 1).items())},
           }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    for path, expected in ((LOCK, LOCK_SHA), (V2, V2_SHA), (CAPS_FILE, CAPS_SHA)):
        if digest(path) != expected:
            raise ValueError(f"Frozen development input changed: {path}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    caps = json.loads(CAPS_FILE.read_text(encoding="utf-8"))["caps"]
    points, tasks = development_data(lock, v2)
    encoding = tiktoken.encoding_for_model(ENCODING_MODEL)
    fit = rows_for_group("fit", points["fit"], tasks, encoding)
    cal = rows_for_group("calibration", points["calibration"], tasks, encoding)
    single_fit = [int(row["framing_residual"]) for row in fit if row["tool_messages"] == 1]
    framing = int(statistics.median(single_fit))
    predictions = {(row["task_id"], row["call_index"]): row["baseline"] + framing
                   for row in cal}
    report = {
        "status": "fit_only_framing_calibration_only_transport_probe",
        "identity_lock_sha256": LOCK_SHA, "development_resolution_sha256": V2_SHA,
        "script_sha256": digest(Path(__file__)),
        "test_trajectory_access": False, "run_level_results_access": False,
        "tokenizer": {"model": ENCODING_MODEL, "name": encoding.name,
                      "tiktoken_version": version("tiktoken")},
        "framing_constant_from_fit_single_tool_median": framing,
        "formula": "I_previous + O_previous + sum(tiktoken(tool message_content.content)) + fit-only framing constant",
        "fit": summary(fit, framing), "calibration": summary(cal, framing),
        "calibration_single_tool": summary([row for row in cal if row["tool_messages"] == 1], framing),
        "calibration_multiple_tool": summary([row for row in cal if row["tool_messages"] > 1], framing),
        "B50_calibration_replay": replay(
            points["calibration"],
            {task_id: tasks[task_id] for task_id in lock["calibration_ids"]},
            caps["0.5"], predictions),
    }
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"sha256": digest(OUT), "framing": framing,
                      "fit": {key: value for key, value in report["fit"].items()
                              if key != "single_tool_residual_counts"},
                      "calibration": {key: value for key, value in report["calibration"].items()
                                      if key != "single_tool_residual_counts"},
                      "B50": {key: value for key, value in report["B50_calibration_replay"].items()
                              if key != "details"}}, indent=2))


if __name__ == "__main__":
    main()
