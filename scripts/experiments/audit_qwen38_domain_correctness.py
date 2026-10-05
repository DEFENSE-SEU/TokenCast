from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


INSTRUCTION = "Write your final answer in answer.md:"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return " ".join(text.split())


def answer_letter(text: object, letters: str) -> str | None:
    if not isinstance(text, str) or not text:
        return None
    first = text.splitlines()[0].strip()
    return first if re.fullmatch(f"[{letters}]", first) else None


def patch_letter(text: object, letters: str) -> str | None:
    if not isinstance(text, str):
        return None
    added = [line[1:].strip() for line in text.splitlines()
             if line.startswith("+") and not line.startswith("+++")]
    return added[0] if added and re.fullmatch(f"[{letters}]", added[0]) else None


def task_text_before_instruction(row: dict) -> str:
    text = row.get("task")
    if not isinstance(text, str) or text.count(INSTRUCTION) != 1:
        raise ValueError(f"missing or repeated task instruction: {row['logical_run_id']}")
    return normalize(text.split(INSTRUCTION, 1)[0])


def validate_task(row: dict, mmlu: dict, longbench: dict) -> tuple[str, str]:
    benchmark, task_id = row["benchmark"], row["task_id"]
    task = task_text_before_instruction(row)
    if benchmark == "mmlupro":
        source_id = int(task_id.removeprefix("mmlupro-"))
        item = mmlu.get(source_id)
        if item is None or task_id != f"mmlupro-{source_id}":
            raise ValueError(f"unmapped MMLU-Pro task: {task_id}")
        question_options = item["question"].strip() + "\n\n" + "\n".join(
            f"{chr(65 + i)}. {option}" for i, option in enumerate(item["options"]))
        if task != normalize(question_options):
            raise ValueError(f"MMLU-Pro question/options mismatch: {row['logical_run_id']}")
        answer = item["answer"]
        context_class = "question_options_exact"
        allowed = "ABCDEFGHIJ"
    else:
        source_id = task_id.removeprefix("longbench-")
        item = longbench.get(source_id)
        if item is None or task_id != f"longbench-{source_id}":
            raise ValueError(f"unmapped LongBench-v2 task: {task_id}")
        question_options = "Question: " + item["question"].strip() + "\n\n" + "\n".join(
            f"{letter}. {item[f'choice_{letter}']}" for letter in "ABCD")
        question_options = normalize(question_options)
        if not task.endswith(question_options):
            raise ValueError(f"LongBench-v2 question/options mismatch: {row['logical_run_id']}")
        context = task[:-len(question_options)].strip()
        original_context = normalize(item["context"])
        if context == original_context:
            context_class = "full_context_exact"
        elif len(context) < len(original_context):
            context_class = "shortened_context"
        else:
            raise ValueError(f"LongBench-v2 context changed without shortening: {row['logical_run_id']}")
        answer = item["answer"]
        allowed = "ABCD"
    if not isinstance(answer, str) or not re.fullmatch(f"[{allowed}]", answer):
        raise ValueError(f"invalid original benchmark key: {task_id}")
    return answer, context_class


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--mmlu-parquet", type=Path, required=True)
    parser.add_argument("--longbench-json", type=Path, required=True)
    parser.add_argument("--answers-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mmlu = {int(row.question_id): row._asdict() for row in
            pd.read_parquet(args.mmlu_parquet).itertuples(index=False)}
    longbench = {item["_id"]: item for item in json.loads(
        args.longbench_json.read_text(encoding="utf-8"))}
    prior_answers = {benchmark: json.loads((args.answers_dir / f"{benchmark}.json").read_text(
        encoding="utf-8")) for benchmark in ("mmlupro", "longbench")}
    archive_path = args.archive / "runs.jsonl.gz"
    audit = []
    seen = set()
    with gzip.open(archive_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not row.get("canonical") or row.get("benchmark") not in prior_answers:
                continue
            logical_id = row["logical_run_id"]
            if logical_id in seen:
                raise ValueError(f"duplicate canonical logical run: {logical_id}")
            seen.add(logical_id)
            benchmark = row["benchmark"]
            key, context_class = validate_task(row, mmlu, longbench)
            old = prior_answers[benchmark].get(row["task_id"])
            if old is not None and old["answer"] != key:
                raise ValueError(f"existing and original benchmark keys disagree: {row['task_id']}")
            letters = "ABCDEFGHIJ" if benchmark == "mmlupro" else "ABCD"
            answer = answer_letter(row.get("answer"), letters)
            patch = patch_letter(row.get("submission_patch"), letters)
            if row["category"] != "completed":
                exclusion = f"status_{row['category']}"
            elif answer is None:
                exclusion = "invalid_answer_first_line"
            elif patch is None:
                exclusion = "missing_or_invalid_patch_first_added_line"
            elif patch != answer:
                exclusion = "answer_patch_mismatch"
            else:
                exclusion = None
            audit.append({"logical_run_id": logical_id, "harness": row["harness"],
                          "benchmark": benchmark, "task_id": row["task_id"],
                          "context_class": context_class, "category": row["category"],
                          "excluded_reason": exclusion, "answer": answer,
                          "patch_answer": patch, "reference_answer": key,
                          "correct": answer == key if exclusion is None else None})
    if len(audit) != 640:
        raise ValueError(f"expected 640 canonical domain runs; found {len(audit)}")
    cells = {}
    harness_totals = {}
    for benchmark in ("mmlupro", "longbench"):
        for harness in ("dsh", "openhands"):
            cohort = [row for row in audit if row["benchmark"] == benchmark and row["harness"] == harness]
            eligible_cohort = [row for row in cohort if row["excluded_reason"] is None]
            harness_totals[f"{benchmark}/{harness}"] = {
                "canonical_runs": len(cohort), "tasks": len({row["task_id"] for row in cohort}),
                "completed_runs": sum(row["category"] == "completed" for row in cohort),
                "correct": sum(row["correct"] for row in eligible_cohort),
                "scorable": len(eligible_cohort),
                "excluded": dict(sorted(Counter(row["excluded_reason"] for row in cohort
                                               if row["excluded_reason"]).items())),
            }
            classes = ("question_options_exact",) if benchmark == "mmlupro" else (
                "full_context_exact", "shortened_context")
            for context_class in classes:
                rows = [row for row in cohort if row["context_class"] == context_class]
                eligible = [row for row in rows if row["excluded_reason"] is None]
                cells[f"{benchmark}/{harness}/{context_class}"] = {
                    "canonical_runs": len(rows), "tasks": len({row["task_id"] for row in rows}),
                    "completed_runs": sum(row["category"] == "completed" for row in rows),
                    "correct": sum(row["correct"] for row in eligible),
                    "scorable": len(eligible),
                    "excluded": dict(sorted(Counter(row["excluded_reason"] for row in rows
                                                   if row["excluded_reason"]).items())),
                }
    report = {"status": "exploratory partial Qwen3.8 archive correctness audit; not PDF Table 7",
              "protocol": {
                  "cohort": "canonical archived MMLU-Pro and LongBench-v2 runs",
                  "mapping": "task ID plus exact question/options after whitespace normalization; original local benchmark answer key",
                  "answer": "answer.md first line must be exactly one allowed option letter and agree with submission.patch first added line",
                  "completion": "category=completed only; censored and failed excluded",
                  "denominators": "correct/completed counts invalid completed answers as wrong; correct/all canonical counts censored as wrong and is a conservative completion-correctness yield; correct/strict scorable is conditional on valid answer and patch",
                  "longbench_context": "full original content after whitespace normalization versus shorter archived content; question/options match both",
                  "unit": "run; repeated tasks and DSH/OpenHands reported separately",
              },
              "inputs_sha256": {"archive": sha256(archive_path),
                                "mmlu_parquet": sha256(args.mmlu_parquet),
                                "longbench_json": sha256(args.longbench_json),
                                "mmlu_existing_answers": sha256(args.answers_dir / "mmlupro.json"),
                                "longbench_existing_answers": sha256(args.answers_dir / "longbench.json")},
              "existing_answer_key_overlap": {benchmark: len({row["task_id"] for row in audit
                                                              if row["benchmark"] == benchmark}
                                                            & prior_answers[benchmark].keys())
                                             for benchmark in ("mmlupro", "longbench")},
              "harness_totals": harness_totals, "cells": cells}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output / "run_audit.jsonl").open("w", encoding="utf-8") as stream:
        for row in audit:
            stream.write(json.dumps(row) + "\n")
    lines = ["# Offline Qwen3.8 domain correctness audit", "",
             "Exploratory archived runs. Correctness requires a strict answer and matching patch. Invalid completed answers count as wrong in the completed denominator. All-canonical includes censored runs as wrong and measures completion plus correctness. No provider calls.", "",
             "| Benchmark | Harness | Correct / completed | Correct / all canonical | Correct / strict scorable | Excluded |",
             "|---|---|---:|---:|---:|---|"]
    for name, cell in harness_totals.items():
        benchmark, harness = name.split("/")
        exclusions = ", ".join(f"{reason}: {count}" for reason, count in cell["excluded"].items()) or "0"
        lines.append(f"| {benchmark} | {harness} | {cell['correct']} / {cell['completed_runs']} | "
                     f"{cell['correct']} / {cell['canonical_runs']} | "
                     f"{cell['correct']} / {cell['scorable']} | {exclusions} |")
    lines += ["", "LongBench-v2 context strata:", "",
             "| Benchmark | Harness | Context | Correct / completed | Correct / all canonical | Correct / strict scorable | Excluded |",
             "|---|---|---|---:|---:|---:|---|"]
    for name, cell in cells.items():
        benchmark, harness, context_class = name.split("/")
        if benchmark != "longbench":
            continue
        exclusions = ", ".join(f"{reason}: {count}" for reason, count in cell["excluded"].items()) or "0"
        lines.append(f"| {benchmark} | {harness} | {context_class} | "
                     f"{cell['correct']} / {cell['completed_runs']} | "
                     f"{cell['correct']} / {cell['canonical_runs']} | "
                     f"{cell['correct']} / {cell['scorable']} | {exclusions} |")
    lines += ["", "Original keys and archived question/options were verified for every canonical run. "
              "The existing domain_answers files cover 0 of 32 MMLU-Pro tasks and 7 of 32 LongBench-v2 tasks; "
              "local original benchmark files provide the audited keys."]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
