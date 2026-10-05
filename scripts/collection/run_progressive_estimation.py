from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.experiments.encode_transcript_windows import (
    call_step,
    load_points,
    load_prefixes,
    read_session,
    transcript_steps,
)
from scripts.experiments.run_four_position_baseline_grid import capture_dirs, capture_run_id
from scripts.experiments.run_related_work_heads import calibrate_in_fold, load_grid_points

REPO = Path(__file__).resolve().parents[2]
CLIENT = REPO / "scripts" / "collection" / "estimate_client.mjs"
NODE_ENV = {"NODE_USE_ENV_PROXY": "1"}

SYSTEM_PROMPT = """You are an evaluation agent for SWE-bench style software engineering tasks.
Given a partial coding-agent rollout, estimate whether the agent can still finish the issue successfully within {max_context_window_tokens} total tokens (input + output), and if yes estimate the remaining total tokens needed from the next turn onward."""

USER_PROMPT = """Based on the provided SWE-bench rollout context, you are given the following:
1. The coding agent has completed {completed_turns} turns.
2. Per-turn token usage so far is: {turn_token_usage_text}.
3. The full task must finish within {max_context_window_tokens} total tokens.

Estimate:
1. Whether the agent can still finish the software issue successfully within {max_context_window_tokens} total tokens.
2. If yes, how many additional tokens (input + output) are still needed from the next turn onward. Return an interval: at least est_low tokens and at most est_high tokens.
3. If no, answer "impossible".
4. Prioritize the can-finish judgment over interval tightness. If the task still looks finishable, keep the interval as tight as possible while still covering the true remaining token budget.

Think about typical SWE-bench costs such as repository inspection, targeted code edits, running validation commands, reading failures, and one or two repair iterations.

Output exactly one of the following:
<think> [Your thoughts, typically 2-4 sentences,less than 200 tokens] </think><answer>[est_low, est_high]</answer>
or
<think> [Your thoughts, typically 2-4 sentences,less than 200 tokens] </think><answer>impossible</answer>"""

CALL_SYSTEM_PROMPT = """You are an evaluation agent for SWE-bench style software engineering tasks.
Given a partial coding-agent rollout, estimate how many output tokens the coding agent will produce in one turn."""

CALL_START_USER_PROMPT = """Based on the provided SWE-bench rollout context, you are given the following:
1. The coding agent has completed {completed_turns} turns.
2. Per-turn token usage so far is: {turn_token_usage_text}.
3. The agent is about to write its next turn.

Estimate how many output tokens the agent's next turn will contain, counting its reasoning, its text and any tool call arguments. Return an interval: at least est_low tokens and at most est_high tokens. Keep the interval as tight as possible while still covering the true number.

Output exactly:
<think> [Your thoughts, typically 2-4 sentences,less than 200 tokens] </think><answer>[est_low, est_high]</answer>"""

IN_CALL_USER_PROMPT = """Based on the provided SWE-bench rollout context, you are given the following:
1. The coding agent has completed {completed_turns} turns.
2. Per-turn token usage so far is: {turn_token_usage_text}.
3. The agent is in the middle of its next turn; the part already written is shown above after "ASSISTANT (in progress):".

Estimate how many more output tokens the agent will produce before this turn ends. Return an interval: at least est_low tokens and at most est_high tokens. Keep the interval as tight as possible while still covering the true number.

Output exactly:
<think> [Your thoughts, typically 2-4 sentences,less than 200 tokens] </think><answer>[est_low, est_high]</answer>"""

CALL_POSITIONS = ("call_start", "in_call_billed")


def per_step_usage(events_path: Path, run_ids: set[str]) -> dict[str, dict[int, dict]]:

    usage: dict[str, dict[int, dict]] = defaultdict(dict)
    current_step: dict[str, int] = {}
    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            run_id = event["event_id"].split(":")[1]
            if run_id not in run_ids:
                continue
            payload = event["payload"]
            if event["event_type"] == "request_built":
                current_step[run_id] = int(payload["source_step"])
            elif event["event_type"] in ("api_completed", "api_failed") and payload.get("usage"):
                step = current_step.get(run_id)
                u = payload["usage"]
                prior = usage[run_id].get(step, {"input": 0, "output": 0, "total": 0})
                usage[run_id][step] = {
                    "input": prior["input"] + int(u.get("input_tokens") or 0),
                    "output": prior["output"] + int(u.get("output_tokens") or 0),
                    "total": prior["total"] + int(u.get("total_tokens") or 0),
                }
    return usage


def render_history(task_text: str, steps: dict[int, str], completed: int) -> str:
    blocks = [f"[1] USER\n{task_text}"]
    for index, step in enumerate(sorted(s for s in steps if s <= completed), start=2):
        blocks.append(f"[{index}] TURN {step}\n{steps[step]}")
    return "\n\n".join(blocks)


def usage_text(usage: dict[int, dict], completed: int) -> str:
    if completed == 0:
        return "None yet."
    return "; ".join(
        f"Turn {s}: input {usage[s]['input']} tokens, output {usage[s]['output']} tokens, total {usage[s]['total']} tokens"
        for s in sorted(usage) if s <= completed
    )


def parse_answer(text: str) -> tuple[bool | None, tuple[int, int] | None]:
    match = re.search(r"<answer>(.*?)</answer>", text, re.S)
    answer = match.group(1).strip() if match else text.strip()
    if answer.lower().startswith("impossible"):
        return False, None
    numbers = re.findall(r"-?\d[\d,]*", answer)
    if len(numbers) >= 2:
        lo, hi = (int(n.replace(",", "")) for n in numbers[:2])
        return True, (min(lo, hi), max(lo, hi))
    return None, None


def latest_parsed(records: list[dict]) -> dict[str, dict]:

    latest: dict[str, dict] = {}
    for record in records:
        if record.get("error") or record.get("can_finish") is None:
            continue
        latest[record["point_id"]] = record
    return latest


def call_model(system: str, user: str, *, model: str, reasoning: str, workdir: Path) -> dict:
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "system.txt").write_text(system, encoding="utf-8")
        (tmp_path / "user.txt").write_text(user, encoding="utf-8")
        out = tmp_path / "out.json"
        proc = subprocess.run(
            ["node", str(CLIENT), "--system", str(tmp_path / "system.txt"), "--user", str(tmp_path / "user.txt"), "--out", str(out), "--model", model, "--reasoning", reasoning],
            capture_output=True, text=True, encoding="utf-8", errors="replace", env={**dict(__import__("os").environ), **NODE_ENV}, timeout=600,
        )
        if not out.exists():
            return {"error": (proc.stderr or proc.stdout)[-2000:], "text": ""}
        return json.loads(out.read_text(encoding="utf-8"))


def build_task_jobs(args, captures, points, usage, run_ids, totals, budget) -> list[dict]:
    by_run_position: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for p in points:
        prefix = p["experiment_id"].split("--")[0]
        if prefix == "task_initialization":
            by_run_position[(p["run_id"], prefix)][0] = p
        elif prefix == "task_update":
            step = call_step(p)
            by_run_position[(p["run_id"], prefix)][step - 1] = p
    jobs = []
    for run_id in run_ids:
        task_text, steps = transcript_steps(read_session(captures[run_id]))
        tu_points = by_run_position.get((run_id, "task_update"), {})
        completed_options = [0] + sorted(k for k in tu_points if k > 0)
        sampled = [0] + [k for k in completed_options[1:] if (k - 1) % args.every == 0][: args.max_turns]
        for completed in sampled:
            point = by_run_position.get((run_id, "task_initialization"), {}).get(0) if completed == 0 else tu_points.get(completed)
            if point is None:
                continue
            spent = sum(usage[run_id][s]["total"] for s in usage[run_id] if s <= completed)
            actual_remaining = totals[run_id] - spent
            system = SYSTEM_PROMPT.format(max_context_window_tokens=budget)
            user = render_history(task_text, steps, completed) + "\n\n" + USER_PROMPT.format(
                completed_turns=completed, turn_token_usage_text=usage_text(usage[run_id], completed), max_context_window_tokens=budget,
            )
            jobs.append({"run_id": run_id, "task_id": point["task_id"], "completed_turns": completed, "point_id": point["point_id"], "experiment_id": point["experiment_id"], "fold": point["fold"], "spent": spent, "actual_remaining": actual_remaining, "actual_total": totals[run_id], "target_value": point["target_value"], "budget": budget, "system": system, "user": user})
    return jobs


def build_call_jobs(args, captures, points, usage, run_ids, budget) -> list[dict]:

    by_run: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for p in points:
        prefix = p["experiment_id"].split("--")[0]
        if prefix in CALL_POSITIONS:
            by_run[p["run_id"]][prefix].append(p)
    prefixes = load_prefixes(args.events, set(run_ids))
    jobs = []
    for run_id in run_ids:
        task_text, steps = transcript_steps(read_session(captures[run_id]))
        call_points = sorted(by_run[run_id]["call_start"], key=lambda p: call_step(p) or 0)
        in_call_points = sorted(by_run[run_id]["in_call_billed"], key=lambda p: p["source_event_id"])
        sampled = [("call_start", p) for p in call_points[:: args.every][: args.max_turns]]
        sampled += [("in_call", p) for p in in_call_points[:: args.every][: args.max_turns]]
        for kind, point in sampled:
            completed = (call_step(point) or 1) - 1
            history = render_history(task_text, steps, completed)
            fields = {"completed_turns": completed, "turn_token_usage_text": usage_text(usage[run_id], completed)}
            if kind == "call_start":
                user = history + "\n\n" + CALL_START_USER_PROMPT.format(**fields)
            else:
                user = (history + "\n\nASSISTANT (in progress):\n" + prefixes.get(point["source_event_id"], "")
                        + "\n\n" + IN_CALL_USER_PROMPT.format(**fields))
            jobs.append({"kind": kind, "run_id": run_id, "task_id": point["task_id"], "completed_turns": completed, "point_id": point["point_id"], "experiment_id": point["experiment_id"], "fold": point["fold"], "target_value": point["target_value"], "budget": budget, "system": CALL_SYSTEM_PROMPT, "user": user})
    return jobs


def write_split_calibration(grid: Path, output: Path, predictions: list[dict]) -> None:

    by_point = {p["point_id"]: p for p in predictions}
    for fit_dir in sorted(grid.glob("split-*/fit-*")):
        out_dir = output / fit_dir.parent.name / fit_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as sink:
            for rows in load_grid_points(fit_dir).values():
                rows = [r for r in rows if r["point_id"] in by_point]
                if not rows:
                    continue
                calibrated = calibrate_in_fold(rows, {r["point_id"]: by_point[r["point_id"]]["prediction"] for r in rows})["calibrated"]
                for r in rows:
                    keys = {k: r[k] for k in ("point_id", "experiment_id", "run_id", "task_id", "fold", "target_value", "sample_weight")}
                    sink.write(json.dumps({**by_point[r["point_id"]], **keys}) + "\n")
                    if r["point_id"] in calibrated:
                        value, lower, upper = calibrated[r["point_id"]]
                        sink.write(json.dumps({**keys, "candidate_id": "progressive_interval_calibrated", "prediction": value,
                                               "lower": lower, "upper": upper, "label_status": "observed"}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grid", type=Path)
    parser.add_argument("--runs-root", type=Path, action="append", required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning", default="low")
    parser.add_argument("--every", type=int, default=2, help="sample every k-th completed Call")
    parser.add_argument("--max-turns", type=int, default=12, help="at most this many sampled Calls per run")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit-runs", type=int)
    parser.add_argument("--call-level", action="store_true", help="ask about the next turn and the turn in progress")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    captures = {capture_run_id(c): c for root in args.runs_root for c in capture_dirs(root)}
    fit_dir = sorted(args.grid.glob("split-*/fit-*"))[0]
    points = [p for p in load_points(fit_dir) if p["run_id"] in captures]
    run_ids = sorted({p["run_id"] for p in points})
    if args.limit_runs:
        run_ids = run_ids[: args.limit_runs]
    usage = per_step_usage(args.events, set(run_ids))
    totals = {r: sum(u["total"] for u in usage[r].values()) for r in run_ids}
    budget = int(statistics.median(totals.values()))

    if args.call_level:
        jobs = build_call_jobs(args, captures, points, usage, run_ids, budget)
    else:
        jobs = build_task_jobs(args, captures, points, usage, run_ids, totals, budget)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} prompts over {len(run_ids)} runs; budget {budget}; median prompt chars {statistics.median(len(j['user']) for j in jobs):.0f}", flush=True)
    if args.dry_run:
        (args.output / "sample_prompt.txt").write_text(jobs[min(3, len(jobs) - 1)]["system"] + "\n\n=====\n\n" + jobs[min(3, len(jobs) - 1)]["user"], encoding="utf-8")
        return

    done_path = args.output / "estimates.jsonl"
    records = [json.loads(line) for line in done_path.open(encoding="utf-8")] if done_path.exists() else []
    finished = set(latest_parsed(records))
    pending = [j for j in jobs if j["point_id"] not in finished]
    print(f"{len(pending)} pending, {len(finished)} already done", flush=True)
    workdir = args.output / "tmp"
    workdir.mkdir(exist_ok=True)

    def run_job(job):
        started = time.time()
        reply = call_model(job["system"], job["user"], model=args.model, reasoning=args.reasoning, workdir=workdir)
        can_finish, interval = parse_answer(reply.get("text", ""))
        return {**{k: v for k, v in job.items() if k not in ("system", "user")}, "model": args.model, "reasoning": args.reasoning, "reply": reply.get("text", ""), "error": reply.get("error"), "usage": reply.get("usage"), "elapsed_s": time.time() - started, "can_finish": can_finish, "interval": interval}

    if pending:
        with done_path.open("a", encoding="utf-8") as sink, ThreadPoolExecutor(args.concurrency) as pool:
            futures = [pool.submit(run_job, job) for job in pending]
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                sink.flush()
                if index % 20 == 0 or index == len(futures):
                    print(f"{index}/{len(futures)} replies", flush=True)

    records = [json.loads(line) for line in done_path.open(encoding="utf-8")]
    rows = list(latest_parsed(records).values())
    predictions = []
    for r in rows:
        if r["interval"] is not None:
            lower, upper = r["interval"]
            point = (lower + upper) / 2
        elif args.call_level:
            continue
        else:
            lower = upper = point = max(0, r["budget"] - r["spent"])
        predictions.append({"candidate_id": "progressive_interval", "point_id": r["point_id"], "experiment_id": r["experiment_id"], "run_id": r["run_id"], "task_id": r["task_id"], "fold": r["fold"], "prediction": point, "lower": lower, "upper": upper, "target_value": r["target_value"], "sample_weight": None, "label_status": "observed", "latency_ms": 1000 * r["elapsed_s"], "estimator_model": r["model"]})
    (args.output / "predictions.jsonl").write_text("\n".join(json.dumps(p) for p in predictions) + "\n", encoding="utf-8")
    write_split_calibration(args.grid, args.output, predictions)
    attempted = {r["point_id"] for r in records}
    summary = {
        "model": args.model, "reasoning": args.reasoning, "budget": budget, "call_level": args.call_level,
        "n_prompts": len(attempted), "n_parsed": len(rows), "n_exported": len(predictions),
        "n_impossible": sum(r["can_finish"] is False for r in rows),
        "n_unresolved": len(attempted) - len(rows),
        "mean_elapsed_s": statistics.fmean(r["elapsed_s"] for r in rows) if rows else None,
        "estimation_tokens_total": sum((r.get("usage") or {}).get("totalTokens", 0) for r in records),
    }
    with_interval = [r for r in rows if r["interval"]]
    if args.call_level:
        summary["interval_coverage"] = statistics.fmean((r["interval"][0] <= r["target_value"] <= r["interval"][1]) for r in with_interval) if with_interval else None
    else:
        summary["can_finish_accuracy"] = statistics.fmean(((r["actual_total"] <= r["budget"]) == r["can_finish"]) for r in rows) if rows else None
        summary["interval_coverage"] = statistics.fmean((r["interval"][0] <= r["actual_remaining"] <= r["interval"][1]) for r in with_interval) if with_interval else None
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
