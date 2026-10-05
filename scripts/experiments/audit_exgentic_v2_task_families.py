from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from scripts.experiments.run_public_trajectory_validation import _sha256


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "workspace/experiments/exgentic_v2_feasibility_20260923"
DATA = ROOT / "workspace/external/exgentic_agent_llm_traces_v2_4b8ad4a"
LOCK = AUDIT / "source_lock.json"
RECEIPT = AUDIT / "download_receipt.json"
OUTPUT = AUDIT / "task_family_audit.json"
COLUMNS = [
    "benchmark", "harness", "models", "run_id", "session_id", "total_tokens",
    "spans.list.element.start_time", "spans.list.element.status.code",
    "spans.list.element.attributes.gen_ai.usage.input_tokens",
    "spans.list.element.attributes.gen_ai.usage.output_tokens",
    "spans.list.element.attributes.gen_ai.input.messages",
    "spans.list.element.attributes.gen_ai.system_instructions",
]


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\u2018", "'").replace("\u2019", "'")
    value = value.replace("\u201c", '"').replace("\u201d", '"')
    return " ".join(value.split()).strip().lower()


def _texts(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        document = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(document, dict):
        document = [document]
    if not isinstance(document, list):
        return []
    result = []
    for item in document:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("content"), str):
            result.append(item["content"])
        parts = item.get("parts")
        if isinstance(parts, list):
            result.extend(part["content"] for part in parts
                          if isinstance(part, dict) and isinstance(part.get("content"), str))
    return result


def _initial_non_policy_user_text(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        messages = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        for part in message.get("parts") or []:
            if not isinstance(part, dict) or not isinstance(part.get("content"), str):
                continue
            value = part["content"]
            if ("You are a customer service agent" not in value
                    and "<policy>" not in value and len(value.strip()) >= 8):
                return _normalize(value)
    return None


def _extract(benchmark: str, texts: list[str], first_input: str | None) -> str | None:
    if benchmark == "appworld":
        for content in texts:
            marker = "Task from supervisor:"
            if marker in content:
                return _normalize(content.rsplit(marker, 1)[1])
    elif benchmark == "browsecompplus":
        for content in texts:
            match = re.search(r"\bQuestion:\s*(.*?)\s*\nNote:", content, re.DOTALL)
            if match:
                return _normalize(match.group(1))
    elif benchmark == "swebench":
        for content in texts:
            match = re.search(r"## Issue to resolve:\s*(.*?)\s*## Execution Environment", content, re.DOTALL)
            if match:
                return _normalize(match.group(1))
    elif benchmark.startswith("tau2_"):
        for content in texts:
            match = re.search(r"First Observation:\s*sender=['\"]user['\"]\s+message=(.*?)(?:\n|$)", content)
            if match:
                value = match.group(1).strip().strip("'\"")
                return _normalize(value)
        return _initial_non_policy_user_text(first_input)
    return None


def _clean(spans: list[dict], total_tokens: int | None) -> bool:
    if len(spans) < 2:
        return False
    stamps = []
    cost = 0
    for span in spans:
        if (span.get("status") or {}).get("code") != 1:
            return False
        stamp = span.get("start_time")
        if stamp is None:
            return False
        stamps.append(stamp)
        attrs = span.get("attributes") or {}
        inputs = attrs.get("gen_ai.usage.input_tokens")
        outputs = attrs.get("gen_ai.usage.output_tokens")
        if not (type(inputs) is int and type(outputs) is int
                and inputs >= 0 and outputs >= 0 and inputs + outputs > 0):
            return False
        cost += inputs + outputs
    return stamps == sorted(stamps) and cost == total_tokens


def main() -> int:
    if OUTPUT.exists():
        raise ValueError("task family audit exists; refusing overwrite")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    if receipt["revision"] != lock["revision"] or len(receipt["files"]) != 9:
        raise ValueError("incomplete pinned source receipt")
    for expected, actual in zip(lock["files"], receipt["files"]):
        if expected["path"] != actual["path"] or expected["sha256"] != actual["sha256"]:
            raise ValueError("receipt differs from lock")

    rows = 0
    clean_rows = 0
    coverage = defaultdict(Counter)
    families: dict[str, dict] = {}
    missing = Counter()
    for source_info in lock["files"]:
        source = DATA / source_info["path"]
        if source.stat().st_size != source_info["size"] or _sha256(source) != source_info["sha256"]:
            raise ValueError(f"source SHA256 differs: {source}")
        reader = pq.ParquetFile(source)
        for batch in reader.iter_batches(batch_size=2, columns=COLUMNS):
            for row in batch.to_pylist():
                rows += 1
                benchmark = row["benchmark"]
                harness = row["harness"]
                model = row["models"][0]
                cell = (benchmark, model, harness)
                spans = row["spans"] or []
                clean = _clean(spans, row["total_tokens"])
                clean_rows += clean
                coverage[cell]["sessions"] += 1
                coverage[cell]["clean_sessions"] += clean
                first = next((span for span in spans if (span.get("status") or {}).get("code") == 1), None)
                if first is None:
                    missing[(benchmark, "no_successful_span")] += 1
                    continue
                attrs = first.get("attributes") or {}
                texts = _texts(attrs.get("gen_ai.system_instructions"))
                texts.extend(_texts(attrs.get("gen_ai.input.messages")))
                task = _extract(benchmark, texts, attrs.get("gen_ai.input.messages"))
                if not task:
                    missing[(benchmark, "no_task_text_marker")] += 1
                    continue
                coverage[cell]["task_text_sessions"] += 1
                coverage[cell]["clean_task_text_sessions"] += clean
                coverage[cell]["clean_task_text_prefixes"] += (len(spans) - 1) if clean else 0
                key = benchmark + ":" + _hash(task)
                group = families.setdefault(key, {
                    "benchmark": benchmark, "rows": 0, "clean_rows": 0,
                    "models": set(), "harnesses": set(), "run_ids": set(),
                    "full_prompt_hashes": set(),
                })
                group["rows"] += 1
                group["clean_rows"] += clean
                group["models"].add(model)
                group["harnesses"].add(harness)
                group["run_ids"].add(row["run_id"])
                group["full_prompt_hashes"].add(_hash("\n".join(texts)))
    if rows != lock["hub_size_api_rows"]:
        raise ValueError("row count differs from source lock")
    cells = [
        {"benchmark": b, "model": m, "harness": h, **dict(counts)}
        for (b, m, h), counts in sorted(coverage.items())
    ]
    by_benchmark = defaultdict(Counter)
    for cell in cells:
        by_benchmark[cell["benchmark"]].update({
            name: cell.get(name, 0)
            for name in ("sessions", "clean_sessions", "task_text_sessions",
                         "clean_task_text_sessions", "clean_task_text_prefixes")
        })
    group_summary = defaultdict(Counter)
    for group in families.values():
        bucket = group_summary[group["benchmark"]]
        bucket["families"] += 1
        bucket["duplicate_families"] += group["rows"] > 1
        bucket["families_crossing_models"] += len(group["models"]) > 1
        bucket["families_crossing_harnesses"] += len(group["harnesses"]) > 1
        bucket["families_crossing_run_ids"] += len(group["run_ids"]) > 1
        bucket["families_with_distinct_full_prompts"] += len(group["full_prompt_hashes"]) > 1
        bucket["max_family_size"] = max(bucket["max_family_size"], group["rows"])
        bucket["clean_families"] += group["clean_rows"] > 0
    report = {
        "dataset": lock["dataset"],
        "revision": lock["revision"],
        "rows": rows,
        "clean_rows": clean_rows,
        "task_text_extractor": "benchmark-specific marker and normalized explicit issue/question/supervisor task/first observation",
        "normalization": "Unicode NFKC, curly quote folding, lowercasing, whitespace collapse",
        "task_group_key": "benchmark + ':' + SHA256(normalized task text)",
        "source_identity_limit": "Text equivalence may merge hidden-state variants; paraphrases may remain split. No explicit benchmark task ID exists.",
        "coverage_by_benchmark": {key: dict(value) for key, value in sorted(by_benchmark.items())},
        "family_summary_by_benchmark": {key: dict(value) for key, value in sorted(group_summary.items())},
        "missing_reasons": {f"{key[0]}:{key[1]}": value for key, value in sorted(missing.items())},
        "cells": cells,
        "verified_turns_read": False,
        "forecast_labels_read": False,
        "forecast_scores_computed": False,
    }
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"rows": rows, "clean_rows": clean_rows,
                      "coverage_by_benchmark": report["coverage_by_benchmark"],
                      "family_summary_by_benchmark": report["family_summary_by_benchmark"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
