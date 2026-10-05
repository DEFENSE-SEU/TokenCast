from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


from scripts.collection.run_progressive_estimation import call_model, render_history
from scripts.experiments.encode_transcript_windows import (
    call_step,
    load_points,
    load_prefixes,
    read_session,
    transcript_steps,
)
from scripts.experiments.run_four_position_baseline_grid import capture_dirs, capture_run_id
from scripts.experiments.run_related_work_heads import calibrate_in_fold, load_grid_points

SYSTEM_PROMPT = """You are the coding agent working on a SWE-bench style software engineering task.
Before you write anything, judge how long your own output will be.
Answer with a single integer inside <answer></answer> tags and nothing else."""

CALL_START_PROMPT = """You are about to send your next message in this session.
Estimate the number of output tokens that message will contain, counting your reasoning,
your text and any tool call arguments.
Answer with <answer>N</answer>."""

IN_CALL_PROMPT = """You are in the middle of writing your next message. The part you have already
produced is shown above after "ASSISTANT (in progress):".
Estimate the number of output tokens still to come before this message ends.
Answer with <answer>N</answer>."""

IN_CALL_POSITIONS = ("in_call_visible", "in_call_billed")


def parse_length(text: str) -> int | None:
    match = re.search(r"<answer>\s*([0-9][0-9,_ ]*)\s*</answer>", text or "", re.IGNORECASE)
    if not match:
        match = re.search(r"\b([0-9][0-9,_]{1,8})\b", text or "")
    if not match:
        return None
    value = int(re.sub(r"[,_ ]", "", match.group(1)))
    return value if value > 0 else None


def build_jobs(args, captures, points) -> list[dict]:

    by_run: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for p in points:
        prefix = p["experiment_id"].split("--")[0]
        if prefix in ("call_start",) + IN_CALL_POSITIONS:
            by_run[p["run_id"]][prefix].append(p)
    prefixes = load_prefixes(args.events, set(by_run))
    jobs = []
    for run_id, by_position in sorted(by_run.items()):
        task_text, steps = transcript_steps(read_session(captures[run_id]))

        call_points = sorted(by_run[run_id]["call_start"], key=lambda p: call_step(p) or 0)
        for point in call_points[:: args.every][: args.max_turns]:
            step = call_step(point) or 1
            jobs.append({
                "kind": "call_start", "run_id": run_id, "task_id": point["task_id"], "cutoff": f"call:{step}",
                "points": [{"point_id": point["point_id"], "experiment_id": point["experiment_id"]}],
                "system": SYSTEM_PROMPT,
                "user": render_history(task_text, steps, step - 1) + "\n\n" + CALL_START_PROMPT,
            })

        in_call: dict[str, dict[str, dict]] = defaultdict(dict)
        for position in IN_CALL_POSITIONS:
            for point in by_position.get(position, []):
                in_call[point["source_event_id"]][position] = point
        for event_id in sorted(in_call)[:: args.every][: args.max_turns]:
            group = in_call[event_id]
            sample = next(iter(group.values()))
            step = call_step(sample) or 1
            history = render_history(task_text, steps, step - 1)
            jobs.append({
                "kind": "in_call", "run_id": run_id, "task_id": sample["task_id"], "cutoff": event_id,
                "points": [{"point_id": p["point_id"], "experiment_id": p["experiment_id"]} for p in group.values()],
                "system": SYSTEM_PROMPT,
                "user": history + "\n\nASSISTANT (in progress):\n" + prefixes.get(event_id, "") + "\n\n" + IN_CALL_PROMPT,
            })
    return jobs


def write_predictions(args, answers: dict[str, float]) -> list[dict]:

    summary = []
    for fit_dir in sorted(args.grid.glob("split-*/fit-*")):
        out_dir = args.output / fit_dir.parent.name / fit_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        sink = (out_dir / "predictions.jsonl").open("w", encoding="utf-8")
        for experiment, rows in load_grid_points(fit_dir).items():
            prefix = experiment.split("--")[0]
            if prefix not in ("call_start",) + IN_CALL_POSITIONS:
                continue
            rows = [r for r in rows if r["point_id"] in answers]
            if not rows:
                continue
            estimates = calibrate_in_fold(rows, {r["point_id"]: answers[r["point_id"]] for r in rows})
            for name in ("raw", "calibrated"):
                errors: dict[str, list[float]] = defaultdict(list)
                for r in rows:
                    pid = r["point_id"]
                    if pid not in estimates[name]:
                        continue
                    value, lower, upper = estimates[name][pid]
                    sink.write(json.dumps({
                        **{k: r[k] for k in ("point_id", "experiment_id", "run_id", "task_id", "fold", "target_value", "sample_weight")},
                        "candidate_id": f"response_length_prompt_{name}", "prediction": value,
                        "lower": lower, "upper": upper, "label_status": "observed",
                    }) + "\n")
                    errors[r["task_id"]].append(abs(value - float(r["target_value"])))
                if errors:
                    summary.append({
                        "split": fit_dir.parent.name, "position": prefix, "candidate": f"response_length_prompt_{name}",
                        "mae": statistics.fmean(statistics.fmean(v) for v in errors.values()), "points": sum(len(v) for v in errors.values()),
                    })
        sink.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grid", type=Path)
    parser.add_argument("--runs-root", type=Path, action="append", required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--reasoning", default="low")
    parser.add_argument("--every", type=int, default=2, help="sample every k-th cutoff of a run")
    parser.add_argument("--max-turns", type=int, default=12, help="at most this many cutoffs per run and position")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit-runs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    captures = {capture_run_id(c): c for root in args.runs_root for c in capture_dirs(root)}
    fit_dir = sorted(args.grid.glob("split-*/fit-*"))[0]
    points = [p for p in load_points(fit_dir) if p["run_id"] in captures]
    if args.limit_runs:
        keep = sorted({p["run_id"] for p in points})[: args.limit_runs]
        points = [p for p in points if p["run_id"] in keep]
    jobs = build_jobs(args, captures, points)
    args.output.mkdir(parents=True, exist_ok=True)
    covered = sum(len(j["points"]) for j in jobs)
    print(f"{len(jobs)} prompts covering {covered} points; median prompt chars {statistics.median(len(j['user']) for j in jobs):.0f}", flush=True)
    if args.dry_run:
        (args.output / "sample_prompt.txt").write_text(jobs[0]["system"] + "\n\n=====\n\n" + jobs[0]["user"][-4000:], encoding="utf-8")
        return

    done_path = args.output / "estimates.jsonl"


    finished = set()
    if done_path.exists():
        for line in done_path.open(encoding="utf-8"):
            record = json.loads(line)
            if record.get("answer"):
                finished.add(record["cutoff_key"])
    pending = [j for j in jobs if f"{j['run_id']}|{j['cutoff']}" not in finished]
    print(f"{len(pending)} pending, {len(finished)} already done", flush=True)
    workdir = args.output / "tmp"
    workdir.mkdir(exist_ok=True)

    def run_job(job):
        started = time.time()
        reply = call_model(job["system"], job["user"], model=args.model, reasoning=args.reasoning, workdir=workdir)
        return {
            "cutoff_key": f"{job['run_id']}|{job['cutoff']}", "kind": job["kind"], "run_id": job["run_id"], "task_id": job["task_id"],
            "points": job["points"], "model": args.model, "reasoning": args.reasoning,
            "reply": reply.get("text", ""), "answer": parse_length(reply.get("text", "")), "error": reply.get("error"),
            "usage": reply.get("usage"), "elapsed_s": time.time() - started,
        }

    if pending:
        with done_path.open("a", encoding="utf-8") as sink, ThreadPoolExecutor(args.concurrency) as pool:
            futures = [pool.submit(run_job, job) for job in pending]
            for index, future in enumerate(as_completed(futures), start=1):
                sink.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
                sink.flush()
                if index % 20 == 0 or index == len(futures):
                    print(f"{index}/{len(futures)} replies", flush=True)

    rows = [json.loads(line) for line in done_path.open(encoding="utf-8")]
    latest = {r["cutoff_key"]: r for r in rows}
    rows = list(latest.values())
    answers = {p["point_id"]: float(r["answer"]) for r in rows if r["answer"] for p in r["points"]}
    summary_rows = write_predictions(args, answers)
    summary = {
        "model": args.model, "reasoning": args.reasoning, "n_prompts": len(rows),
        "n_parsed": sum(bool(r["answer"]) for r in rows), "n_errors": sum(bool(r.get("error")) for r in rows),
        "points_covered": len(answers),
        "mean_elapsed_s": statistics.fmean(r["elapsed_s"] for r in rows),
        "estimation_tokens_total": sum((r.get("usage") or {}).get("totalTokens", 0) for r in rows),
        "mae_by_position": summary_rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "mae_by_position"}, indent=1), flush=True)
    for row in summary_rows:
        print(f"{row['split']} {row['position']} {row['candidate']}: mae {row['mae']:.4g} on {row['points']} points", flush=True)


if __name__ == "__main__":
    main()
