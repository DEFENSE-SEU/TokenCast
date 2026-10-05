from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import tiktoken

from .develop_envbench_jvm_budget_models import (
    BASE, CAPS_FILE, CAPS_SHA, LOCK, LOCK_SHA, ROOT, V2, V2_SHA,
    development_data, digest, read_jsonl,
)


OUT = ROOT / "workspace/experiments/envbench_transport_input_dev_20260923/report.json"
ENCODING = "o200k_base"


def extract(group: str, ids: set[str], tasks: dict[str, dict], encoder) -> list[dict]:
    calls = read_jsonl(BASE / f"development_{group}_calls_v2.jsonl")
    by_id = defaultdict(list)
    for call in calls:
        if call["task_id"] not in ids:
            raise ValueError("Exported call outside frozen development split")
        by_id[call["task_id"]].append(call)
    if set(by_id) != ids:
        raise ValueError("Missing call export")
    output = []
    for task_id in sorted(ids):
        raw_path = BASE / f"raw_{group}" / f"{task_id}.jsonl"
        raw = raw_path.read_bytes()
        if sha256(raw).hexdigest() != tasks[task_id]["raw_sha256"]:
            raise ValueError("Raw development task hash changed")
        positions = {}
        tool_contents = []
        for event_index, line in enumerate(line for line in raw.splitlines() if line.strip()):
            event = json.loads(line)
            for message_index, message in enumerate(event.get("messages", [])):
                if not isinstance(message, dict):
                    continue
                kind = message.get("message_type")
                if kind == "tool":
                    wrapper = message.get("message_content")
                    if not isinstance(wrapper, dict) or not isinstance(wrapper.get("content"), str):
                        raise ValueError("Non-string prior tool content")
                    tool_contents.append(wrapper["content"])
                elif kind == "ai":
                    positions[(event_index, message_index)] = list(tool_contents)
                    tool_contents.clear()
        ordered = sorted(by_id[task_id], key=lambda item: item["call_index"])
        if [item["call_index"] for item in ordered] != list(range(len(ordered))):
            raise ValueError("Noncontiguous calls")
        for j, current in enumerate(ordered):
            if j == 0:
                continue
            key = (current["event_index"], current["message_index"])
            contents = positions[key]
            if len(contents) != current["tool_messages_since_previous_ai"]:
                raise ValueError("Tool count differs from exported causal span")
            tool_tokens = sum(len(encoder.encode(text, disallowed_special=()))
                              for text in contents)
            previous = ordered[j - 1]
            base = previous["input_tokens"] + previous["output_tokens"] + tool_tokens
            output.append({
                "task_id": task_id, "call_index": j,
                "tool_messages": len(contents), "prior_tool_content_tokens": tool_tokens,
                "previous_input": previous["input_tokens"],
                "previous_output": previous["output_tokens"],
                "observed_next_input": current["input_tokens"],
                "transport_base": base,
                "residual": current["input_tokens"] - base,
                "spent": current["spent_before"],
            })
    return output


def describe(rows: list[dict], offset: int | None = None) -> dict:
    residual = np.asarray([row["residual"] for row in rows], dtype=int)
    if not len(rows):
        return {"points": 0}
    result = {
        "points": len(rows), "tool_count_distribution": dict(Counter(row["tool_messages"]
                                                               for row in rows)),
        "residual_mode_counts": {str(key): count for key, count in
                                 Counter(residual.tolist()).most_common(12)},
        "residual_median": float(np.median(residual)),
        "residual_min": int(residual.min()), "residual_max": int(residual.max()),
    }
    if offset is not None:
        errors = np.abs(residual - offset)
        targets = np.asarray([row["observed_next_input"] for row in rows], dtype=float)
        result.update({
            "fit_offset": offset, "mae_tokens": float(np.mean(errors)),
            "median_abs_error_tokens": float(np.median(errors)),
            "wape": float(sum(errors) / sum(targets)),
            "fraction_exact": float(np.mean(errors == 0)),
            "fraction_within_2_tokens": float(np.mean(errors <= 2)),
            "fraction_within_10_tokens": float(np.mean(errors <= 10)),
        })
    return result


def main() -> None:
    for path, expected in ((LOCK, LOCK_SHA), (V2, V2_SHA), (CAPS_FILE, CAPS_SHA)):
        if digest(path) != expected:
            raise ValueError(f"Frozen development input changed: {path}")
    if OUT.exists():
        raise FileExistsError(OUT)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    _, tasks = development_data(lock, v2)
    encoder = tiktoken.get_encoding(ENCODING)
    fit = extract("fit", set(lock["fit_ids"]), tasks, encoder)
    cal = extract("calibration", set(lock["calibration_ids"]), tasks, encoder)
    if len(fit) != 4759 or len(cal) != 1502:
        raise ValueError("Development prefix count drift")
    fit_offset = int(np.median([row["residual"] for row in fit]))
    if fit_offset != 9:
        raise ValueError("Fit-only transport offset changed")
    cap = json.loads(CAPS_FILE.read_text(encoding="utf-8"))["caps"]["0.5"]
    report = {
        "status": "fit_calibration_only_transport_input_audit",
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes": {str(path.relative_to(ROOT)): expected for path, expected in (
            (LOCK, LOCK_SHA), (V2, V2_SHA), (CAPS_FILE, CAPS_SHA))},
        "test_trajectory_access": False, "run_level_results_access": False,
        "tokenizer": {"package": "tiktoken", "version": tiktoken.__version__,
                      "encoding": ENCODING},
        "causal_contract": "At current AI arrival, previous settled input/output plus tokenized content of tool messages serialized since previous AI; current AI usage is evaluation label only",
        "formula": "transported_next_input=I_previous+O_previous+sum(o200k_base(tool.message_content.content))+fit_median_residual",
        "fit_only_offset": fit_offset,
        "B50": cap,
        "fit": {"all": describe(fit, fit_offset),
                "single_tool": describe([row for row in fit if row["tool_messages"] == 1], fit_offset),
                "multiple_tools": describe([row for row in fit if row["tool_messages"] > 1], fit_offset)},
        "calibration": {"all": describe(cal, fit_offset),
                        "single_tool": describe([row for row in cal if row["tool_messages"] == 1], fit_offset),
                        "multiple_tools": describe([row for row in cal if row["tool_messages"] > 1], fit_offset)},
        "calibration_points": cal,
    }
    OUT.parent.mkdir(parents=True)
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report_sha256": digest(OUT), "fit_offset": fit_offset,
                      "fit": report["fit"], "calibration": report["calibration"]}, indent=2))


if __name__ == "__main__":
    main()
