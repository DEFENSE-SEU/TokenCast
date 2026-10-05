from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from token_prediction.control import (
    ReplayCheckpoint,
    TrajectoryReplayRecord,
    evaluate_abort_policy,
    evaluate_admission_policy,
    evaluate_fixed_cap,
    evaluate_no_control,
    evaluate_oracle,
)
from token_prediction.dataset import PredictionTarget
from token_prediction.estimators import TokenForecast

ITERATIONS = 10_000
SEED = 20260907
MAIN_LEVELS = {"lower": 0.05, "point": 0.5, "upper": 0.95}


@dataclass
class RunLog:
    run_id: str
    task_id: str
    total_tokens: int
    succeeded: bool
    wall_seconds: float
    cost: float | None
    elapsed_at_seq: dict[int, float]


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_runs(path: Path, prices: tuple[float, float, float] | None) -> dict[str, RunLog]:
    runs: dict[str, RunLog] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            execution = record["execution"]
            usage = record.get("usage") or {}
            cost = None
            if prices is not None:
                cost = (
                    (usage.get("input_tokens", 0) - usage.get("cached_input_tokens", 0)) * prices[0]
                    + usage.get("cached_input_tokens", 0) * prices[1]
                    + usage.get("output_tokens", 0) * prices[2]
                ) / 1e6
            runs[record["run_id"]] = RunLog(
                run_id=record["run_id"],
                task_id=record["task"]["task_id"],
                total_tokens=int(execution["cap"]["provider_accounted_tokens"]),
                succeeded=record["terminal"]["status"] == "completed",
                wall_seconds=float(execution["cap"]["wall_seconds"]),
                cost=cost,
                elapsed_at_seq={},
            )
    return runs


def attach_event_clock(path: Path, runs: dict[str, RunLog]) -> None:

    started: dict[str, datetime] = {}
    times: dict[str, list[tuple[int, datetime]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            run_id = event["event_id"].split(":")[1]
            if run_id not in runs:
                continue
            when = _parse_time(event["occurred_at"])
            if event["event_type"] == "task_started":
                started[run_id] = when
            times[run_id].append((int(event["event_seq"]), when))
    for run_id, run in runs.items():
        origin = started.get(run_id)
        if origin is not None:
            run.elapsed_at_seq = {seq: (when - origin).total_seconds() for seq, when in times[run_id]}


def load_forecast_levels(
    fit_dir: Path,
    alpha: float,
    into_launch: dict,
    into_update: dict,
    *,
    side_info: dict[str, tuple[int, int]] | None = None,
    wanted: set[str] | None = None,
    rename: dict[str, str] | None = None,
    point_index: dict | None = None,
) -> None:

    lower_level, upper_level = round(alpha / 2, 3), round(1 - alpha / 2, 3)
    with (fit_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if wanted is not None and not any(f'"candidate_id": "{c}"' in line for c in wanted):
                continue
            record = json.loads(line)
            prefix = record["experiment_id"].split("--")[0]
            if prefix not in ("task_initialization", "task_update"):
                continue
            candidate, run_id = record["candidate_id"], record["run_id"]
            if wanted is not None and candidate not in wanted:
                continue
            if rename is not None:
                candidate = rename.get(candidate, candidate)
            levels = {lower_level: float(record["lower"]), 0.5: float(record["prediction"]), upper_level: float(record["upper"])}

            levels.update({round(float(q), 3): float(v) for q, v in record.get("quantiles", {}).items()})
            if prefix == "task_initialization":
                into_launch[candidate].setdefault(run_id, {}).update(levels)
                if point_index is not None:
                    point_index.setdefault("launch", {})[run_id] = record["point_id"]
                continue
            if "cutoff_event_seq" in record:
                seq, progress = int(record["cutoff_event_seq"]), int(record["progress_known_amount"])
                if side_info is not None:
                    side_info[record["point_id"]] = (seq, progress)
            elif side_info is not None:
                seq, progress = side_info[record["point_id"]]
            else:
                raise KeyError(f"{fit_dir}: task_update row has no cutoff_event_seq and no side_info was given")
            if point_index is not None:
                point_index.setdefault("update", {}).setdefault(run_id, {})[seq] = record["point_id"]
            entry = into_update[candidate].setdefault(run_id, {}).setdefault(seq, (progress, {}))
            entry[1].update(levels)


def has_level(update_forecasts: dict, level: float) -> bool:

    return all(level in levels for seqs in update_forecasts.values() for _, levels in seqs.values())


def sort_levels(level_dicts) -> tuple[int, int]:

    changed = seen = 0
    for levels in level_dicts:
        seen += 1
        keys = sorted(levels)
        values = [levels[k] for k in keys]
        ordered = sorted(values)
        if ordered != values:
            levels.update(zip(keys, ordered))
            changed += 1
    return changed, seen


def build_records(runs, run_ids, launch: dict[str, dict], update: dict[str, dict]) -> list[TrajectoryReplayRecord]:

    records = []
    for run_id in run_ids:
        run = runs[run_id]
        checkpoints = []
        for seq, (spent, levels) in sorted(update.get(run_id, {}).items()):
            checkpoints.append(ReplayCheckpoint(seq, min(spent, run.total_tokens), TokenForecast(point_id=f"{run_id}:{seq}", target=PredictionTarget.TASK_PROVIDER_ACCOUNTED_REMAINING_TOKENS, lower=levels[0.05], point=levels[0.5], upper=levels[0.95])))
        launch_forecast = None
        if run_id in launch:
            lv = launch[run_id]
            launch_forecast = TokenForecast(point_id=f"{run_id}:launch", target=PredictionTarget.TASK_TOTAL_ACCOUNTED_TOKENS, lower=lv[0.05], point=lv[0.5], upper=lv[0.95])
        records.append(TrajectoryReplayRecord(task_id=run.task_id, trajectory_id=run_id, succeeded=run.succeeded, total_tokens=run.total_tokens, checkpoints=tuple(checkpoints), launch_forecast=launch_forecast))
    return records


def decide(
    run: RunLog, *, budget: int, launch: dict | None, update: dict | None,
    level: float, policy: str, predictor_prelaunch: dict | None = None,
    predictor_launch: dict | None = None,
    predictor_update: dict | None = None,
) -> tuple[str, int, int | None, int, float, str]:

    if policy == "no_control":
        return "executed", run.total_tokens, None, 0, 0.0, "no_control"
    if policy == "fixed_cap":
        return ("executed", run.total_tokens, None, 0, 0.0, "completed") if run.total_tokens <= budget else ("aborted", budget, None, 0, 0.0, "agent_cap")
    if policy == "oracle":
        return ("executed", run.total_tokens, None, 0, 0.0, "completed") if run.succeeded and run.total_tokens <= budget else ("refused", 0, None, 0, 0.0, "oracle")
    if policy not in ("admission", "abort", "combined"):
        raise ValueError(f"unknown policy: {policy}")

    predictor_tokens, predictor_seconds = 0, 0.0
    cost, seconds = (predictor_prelaunch or {}).get(run.run_id, (0, 0.0))
    charged = min(cost, budget)
    predictor_tokens += charged
    predictor_seconds += seconds * (charged / cost if cost else 1.0)
    if charged < cost or (predictor_tokens == budget and run.total_tokens > 0):
        return "refused", 0, None, predictor_tokens, predictor_seconds, "predictor_cap"
    if policy in ("admission", "combined"):
        if launch is None or level not in launch:
            raise ValueError(f"admission needs a launch forecast at level {level} for {run.run_id}")
        cost, seconds = (predictor_launch or {}).get(run.run_id, (0, 0.0))
        charged = min(cost, budget - predictor_tokens)
        predictor_tokens += charged
        predictor_seconds += seconds * (charged / cost if cost else 1.0)
        if charged < cost or (predictor_tokens == budget and run.total_tokens > 0):
            return "refused", 0, None, predictor_tokens, predictor_seconds, "predictor_cap"
        if launch[level] + predictor_tokens > budget:
            return "refused", 0, None, predictor_tokens, predictor_seconds, "forecast"
    if policy in ("abort", "combined") and update is not None:
        for seq, (spent, levels) in sorted(update.items()):
            spent = min(spent, run.total_tokens)
            if spent + predictor_tokens >= budget and run.total_tokens > spent:
                return "aborted", budget - predictor_tokens, None, predictor_tokens, predictor_seconds, "agent_cap"
            if spent + predictor_tokens > budget:
                return "aborted", budget - predictor_tokens, None, predictor_tokens, predictor_seconds, "agent_cap"
            cost, seconds = (predictor_update or {}).get(run.run_id, {}).get(seq, (0, 0.0))
            charged = min(cost, budget - spent - predictor_tokens)
            predictor_tokens += charged
            predictor_seconds += seconds * (charged / cost if cost else 1.0)
            if charged < cost or (spent + predictor_tokens == budget and run.total_tokens > spent):
                return "aborted", spent, seq, predictor_tokens, predictor_seconds, "predictor_cap"
            if spent + predictor_tokens + levels[level] > budget:
                return "aborted", spent, seq, predictor_tokens, predictor_seconds, "forecast"
    if run.total_tokens + predictor_tokens > budget:
        return "aborted", budget - predictor_tokens, None, predictor_tokens, predictor_seconds, "agent_cap"
    return "executed", run.total_tokens, None, predictor_tokens, predictor_seconds, "completed"


def elapsed_at(run: RunLog, outcome: str, spent: int, stop_seq: int | None) -> float:
    if outcome == "refused":
        return 0.0
    if outcome == "executed":
        return run.wall_seconds
    if stop_seq is not None and stop_seq in run.elapsed_at_seq:
        return run.elapsed_at_seq[stop_seq]
    return run.wall_seconds * spent / run.total_tokens if run.total_tokens else 0.0


def predictor_charge(run_id: str, *, policy: str, outcome: str, stop_seq: int | None, predictor_launch: dict | None, predictor_update: dict | None) -> tuple[int, float]:

    tokens, seconds = 0, 0.0
    if predictor_launch is not None and policy in ("admission", "combined"):
        t, s = predictor_launch.get(run_id, (0, 0.0))
        tokens += t
        seconds += s
    if predictor_update is not None and policy in ("abort", "combined") and outcome != "refused":
        per_point = predictor_update.get(run_id, {})
        charged = per_point if outcome == "executed" else {seq: v for seq, v in per_point.items() if seq <= stop_seq}
        for t, s in charged.values():
            tokens += t
            seconds += s
    return tokens, seconds


def replay(runs, run_ids, *, budget, policy, launch=None, update=None, level=0.5, predictor_prelaunch=None, predictor_launch=None, predictor_update=None) -> dict:
    per_run = {}
    for run_id in run_ids:
        run = runs[run_id]
        outcome, spent, stop_seq, predictor_tokens, predictor_seconds, stop_reason = decide(
            run, budget=budget, launch=None if launch is None else launch.get(run_id),
            update=None if update is None else update.get(run_id), level=level, policy=policy,
            predictor_prelaunch=predictor_prelaunch, predictor_launch=predictor_launch,
            predictor_update=predictor_update,
        )
        seconds = elapsed_at(run, outcome, spent, stop_seq)
        cost = None if run.cost is None else run.cost * (spent / run.total_tokens if run.total_tokens else 0.0)
        total_spent = spent + predictor_tokens
        if policy != "no_control" and total_spent > budget:
            raise AssertionError(f"{policy} exceeded budget on {run_id}: {total_spent} > {budget}")
        lost_cap_completion = outcome != "executed" and run.succeeded and run.total_tokens <= budget
        per_run[run_id] = {"task_id": run.task_id, "outcome": outcome, "stop_reason": stop_reason, "agent_tokens": spent, "predictor_tokens": predictor_tokens, "spent_tokens": total_spent, "predictor_seconds": predictor_seconds, "seconds": seconds + predictor_seconds, "cost": cost, "completed": outcome == "executed" and run.succeeded and (policy == "no_control" or total_spent <= budget), "false_stop": lost_cap_completion, "overhead_caused_loss": lost_cap_completion and run.total_tokens + predictor_tokens > budget}
    return {"summary": summarize(per_run, budget), "per_run": per_run}


def summarize(per_run: dict, budget: int) -> dict:
    n = len(per_run)
    return {
        "budget": budget, "n_runs": n,
        "agent_tokens": sum(r["agent_tokens"] for r in per_run.values()),
        "predictor_tokens": sum(r["predictor_tokens"] for r in per_run.values()),
        "spent_tokens": sum(r["spent_tokens"] for r in per_run.values()),
        "predictor_seconds": sum(r["predictor_seconds"] for r in per_run.values()),
        "seconds": sum(r["seconds"] for r in per_run.values()),
        "cost": None if any(r["cost"] is None for r in per_run.values()) else sum(r["cost"] for r in per_run.values()),
        "completed_rate": sum(r["completed"] for r in per_run.values()) / n,
        "false_stop_rate": sum(r["false_stop"] for r in per_run.values()) / n,
        "overhead_caused_loss_rate": sum(r.get("overhead_caused_loss", False) for r in per_run.values()) / n,
        "refused": sum(r["outcome"] == "refused" for r in per_run.values()),
        "aborted": sum(r["outcome"] == "aborted" for r in per_run.values()),
        "hard_cap_hits": sum(r.get("stop_reason") in ("agent_cap", "predictor_cap") for r in per_run.values()),
    }


def task_folds(fit_dir: Path, split_name: str) -> dict[str, int]:

    protocol = json.loads((fit_dir / "development_protocol.json").read_text(encoding="utf-8"))
    seed = int(split_name.rsplit("-", 1)[1])
    plan = next(p for p in protocol["outer_plans"] if int(p["seed"]) == seed)
    return {a["task_id"]: int(a["fold"]) for a in plan["assignments"]}


def crossfit_level(level_points: dict[float, dict], cap_per_run: dict, selection_runs: list[str]) -> float:

    cap_done = sum(cap_per_run[r]["completed"] for r in selection_runs)
    scored = [(sum(res["per_run"][r]["completed"] for r in selection_runs), sum(res["per_run"][r]["spent_tokens"] for r in selection_runs), level) for level, res in sorted(level_points.items())]
    eligible = [s for s in scored if s[0] >= cap_done]
    if eligible:
        return min(eligible, key=lambda s: s[1])[2]
    return min(scored, key=lambda s: (-s[0], s[1]))[2]


def crossfit_operating_point(level_points: dict[float, dict], cap_per_run: dict, fold_of_run: dict[str, int], budget: int) -> tuple[dict, dict[int, float]]:

    chosen = {fold: crossfit_level(level_points, cap_per_run, [r for r, f in fold_of_run.items() if f != fold]) for fold in sorted(set(fold_of_run.values()))}
    per_run = {r: level_points[chosen[fold]]["per_run"][r] for r, fold in fold_of_run.items()}
    return {"summary": summarize(per_run, budget), "per_run": per_run}, chosen


def cross_check(records, budget, result, *, policy, level):

    if policy not in ("no_control", "fixed_cap", "oracle"):
        return
    name = {0.05: "lower", 0.5: "point", 0.95: "upper"}.get(level)
    if policy == "no_control":
        reference = evaluate_no_control(records, budget=budget)
    elif policy == "fixed_cap":
        reference = evaluate_fixed_cap(records, budget=budget)
    elif policy == "oracle":
        reference = evaluate_oracle(records, budget=budget)
    elif policy == "admission" and name:
        reference = evaluate_admission_policy(records, budget=budget, decision_input=name)
    elif policy == "abort" and name:
        reference = evaluate_abort_policy(records, budget=budget, decision_input=name)
    else:
        return
    summary = result["summary"]
    if reference.spent_tokens != summary["agent_tokens"] or reference.successes != round(summary["completed_rate"] * summary["n_runs"]):
        raise AssertionError(f"{policy}@{level}@{budget}: replay differs from control/replay_eval.py")


def task_paired(per_run_a: dict, per_run_b: dict, key: str, *, seed: int) -> dict:
    by_task: dict[str, list[float]] = defaultdict(list)
    for run_id, a in per_run_a.items():
        by_task[a["task_id"]].append(float(a[key]) - float(per_run_b[run_id][key]))
    tasks = sorted(by_task)
    diff = np.asarray([statistics.fmean(by_task[t]) for t in tasks])
    draws = np.random.default_rng(seed).integers(0, len(tasks), size=(ITERATIONS, len(tasks)))
    means = diff[draws].mean(axis=1)
    return {"n_tasks": len(tasks), "task_deltas": dict(zip(tasks, map(float, diff))), "delta": float(diff.mean()), "ci_lower": float(np.percentile(means, 2.5)), "ci_upper": float(np.percentile(means, 97.5))}


def combine_task_paired(seed_estimates: list[dict], *, seed: int) -> dict:

    tasks = sorted(seed_estimates[0]["task_deltas"])
    if any(sorted(item["task_deltas"]) != tasks for item in seed_estimates[1:]):
        raise ValueError("split seeds must evaluate the same task ids")
    diff = np.asarray([statistics.fmean(item["task_deltas"][task] for item in seed_estimates) for task in tasks])
    draws = np.random.default_rng(seed).integers(0, len(tasks), size=(ITERATIONS, len(tasks)))
    means = diff[draws].mean(axis=1)
    return {"n_tasks": len(tasks), "delta": float(diff.mean()), "ci_lower": float(np.percentile(means, 2.5)), "ci_upper": float(np.percentile(means, 97.5))}


def forecast_accuracy(grid: Path, main_table_json: Path | None = None) -> dict[str, float]:

    per_seed: dict[str, list[float]] = defaultdict(list)
    for path in sorted(grid.glob("split-*/fit-*/metrics.json")):
        for experiment, candidates in json.loads(path.read_text(encoding="utf-8")).items():
            if not experiment.startswith("task_update"):
                continue
            for candidate, block in candidates.items():
                mae = block.get("metrics", {}).get("mae")
                if mae is not None:
                    per_seed[candidate].append(float(mae))
    accuracy = {candidate: statistics.fmean(values) for candidate, values in per_seed.items()}
    if main_table_json is not None and main_table_json.is_file():
        report = json.loads(main_table_json.read_text(encoding="utf-8"))
        for panel in report.get("panels", {}).values():
            for experiment, block in panel.get("experiments", {}).items():
                if experiment.split("--")[0] != "task_update":
                    continue
                for candidate, arm in block.get("arms", {}).items():
                    if candidate in accuracy:
                        continue
                    mae = arm.get("metrics", {}).get("mae")
                    if mae is not None:
                        accuracy[candidate] = float(mae)
    return accuracy


PROXY_FORWARD_SECONDS = {


    "task_initialization": 0.0969,
    "task_update": 0.8588,
}
TOKENETC_UPDATE_SECONDS = 0.001405
PROXY_MODEL_CANDIDATES = {"trail_last", "egtp_pool", "tie_encoder"}
TOKENETC_CANDIDATES = {"input_output_scaled_interval", "task_update_ours"}
BAGEN_CANDIDATE = "progressive_interval_calibrated"
SELF_PREDICT_CANDIDATE = "self_estimate_gpt54mini_calibrated"


def load_query_costs(path: Path) -> dict[str, tuple[int, float]]:

    out: dict[str, tuple[int, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("error"):
                out[record["point_id"]] = (0, 0.0)
            else:
                out[record["point_id"]] = (int(record["usage"]["totalTokens"]), float(record["elapsed_s"]))
    return out


def load_self_predict_costs(path: Path) -> dict[str, tuple[int, float]]:

    out: dict[str, tuple[int, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            out[record["task_id"]] = (int(record["estimation_tokens"]), float(record["estimation_wall_seconds"]))
    return out


def find_estimates_file(extras: list[Path], marker: str) -> Path | None:

    for extra in extras:
        path = extra / "estimates.jsonl"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            first = json.loads(handle.readline())
        if marker in first:
            return path
    return None


def build_predictor_costs(runs: dict, launch: dict, update: dict, point_index: dict, *, query_costs: dict | None, self_predict_costs: dict | None) -> dict[str, dict]:

    costs: dict[str, dict] = {}
    for candidate in set(launch) | set(update):
        prelaunch_cost: dict[str, tuple[int, float]] = {}
        launch_cost: dict[str, tuple[int, float]] = {}
        update_cost: dict[str, dict[int, tuple[int, float]]] = {}
        if candidate == BAGEN_CANDIDATE and query_costs is not None:
            for run_id in launch.get(candidate, {}):
                point_id = point_index.get("launch", {}).get(run_id)
                launch_cost[run_id] = query_costs.get(point_id, (0, 0.0))
            for run_id, seqs in update.get(candidate, {}).items():
                update_cost[run_id] = {seq: query_costs.get(point_index.get("update", {}).get(run_id, {}).get(seq), (0, 0.0)) for seq in seqs}
        elif candidate == SELF_PREDICT_CANDIDATE and self_predict_costs is not None:
            prelaunch_cost = {run_id: self_predict_costs[run.task_id] for run_id, run in runs.items() if run.task_id in self_predict_costs}
        elif candidate in PROXY_MODEL_CANDIDATES:
            for run_id in launch.get(candidate, {}):
                launch_cost[run_id] = (0, PROXY_FORWARD_SECONDS["task_initialization"])
            for run_id, seqs in update.get(candidate, {}).items():
                update_cost[run_id] = {seq: (0, PROXY_FORWARD_SECONDS["task_update"]) for seq in seqs}
        elif candidate in TOKENETC_CANDIDATES:
            for run_id in launch.get(candidate, {}):
                launch_cost[run_id] = (0, 0.0)
            for run_id, seqs in update.get(candidate, {}).items():
                update_cost[run_id] = {seq: (0, TOKENETC_UPDATE_SECONDS) for seq in seqs}
        else:
            continue
        costs[candidate] = {"prelaunch": prelaunch_cost, "launch": launch_cost, "update": update_cost}
    return costs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grid", type=Path)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level-grid", action="append", default=[], help="<alpha>=<grid_dir> fitted at another interval alpha")
    parser.add_argument("--extra", action="append", default=[], type=Path, help="companion export directory, same <split>/fit-*/predictions.jsonl layout")
    parser.add_argument("--common-update-points", action="store_true", help="restrict Task-update checkpoints to the (run_id, seq) pairs every update candidate has")
    parser.add_argument("--main-table-json", type=Path, help="build_main_table.py report to fall back to for a --extra candidate's Task-update MAE")
    parser.add_argument("--charge-predictor", action="store_true", help="charge recorded prediction tokens and time against each run's hard budget")
    parser.add_argument("--level-selection", choices=("crossfit", "in_sample"), default="crossfit", help="crossfit: each outer test fold replays the level chosen on the other folds' runs at the cap's budget; in_sample: cheapest (level, budget) point over all replayed runs")
    parser.add_argument("--sort-levels", action=argparse.BooleanOptionalAction, default=True, help="sort each forecast's level values into monotone order before any policy reads a level (--no-sort-levels reads them as exported)")
    parser.add_argument("--launch-candidates", default="empirical,lightgbm_generic")
    parser.add_argument("--update-candidates", default="empirical,call_clock_compound,deduct_only,lightgbm_generic,task_update_ours")
    parser.add_argument("--ours-launch", default="lightgbm_generic")
    parser.add_argument("--ours-update", default="task_update_ours")
    parser.add_argument("--budget-quantiles", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--price-input", type=float)
    parser.add_argument("--price-cached", type=float)
    parser.add_argument("--price-output", type=float)
    args = parser.parse_args()

    prices = None
    if args.price_input is not None:
        if args.price_cached is None or args.price_output is None:
            raise SystemExit("all three prices are required for cost accounting")
        prices = (args.price_input, args.price_cached, args.price_output)
    runs = load_runs(args.runs, prices)
    attach_event_clock(args.events, runs)
    launch_ids, update_ids = args.launch_candidates.split(","), args.update_candidates.split(",")
    level_grids = [(0.1, args.grid)] + [(float(item.split("=")[0]), Path(item.split("=")[1])) for item in args.level_grid]
    quantiles = [float(q) for q in args.budget_quantiles.split(",")]
    split_names = [d.parent.name for d in sorted(args.grid.glob("split-*/fit-*"))]
    if not split_names:
        raise SystemExit(f"no split-*/fit-* under {args.grid}")

    extra_wanted = set(launch_ids) | set(update_ids)
    query_costs = self_predict_costs = None
    if args.charge_predictor:
        query_path = find_estimates_file(args.extra, "usage")
        query_costs = load_query_costs(query_path) if query_path else None
        self_predict_path = find_estimates_file(args.extra, "estimation_tokens")
        self_predict_costs = load_self_predict_costs(self_predict_path) if self_predict_path else None

    all_rows: list[dict] = []
    paired: list[dict] = []
    budgets_out = None
    checkpoint_counts: dict[str, dict[str, int]] = {}
    reordered: dict[str, dict[str, dict]] = {}
    hard_cap_audit: list[dict] = []
    for split_name in split_names:
        launch: dict = defaultdict(dict)
        update: dict = defaultdict(dict)
        side_info: dict[str, tuple[int, int]] = {}
        point_index: dict = {}
        main_fit_dirs = list((args.grid / split_name).glob("fit-*"))
        load_forecast_levels(main_fit_dirs[0], 0.1, launch, update, side_info=side_info, point_index=point_index)
        for extra in args.extra:
            fit_dirs = sorted((extra / split_name).glob("fit-*"))
            if not fit_dirs:
                raise SystemExit(f"{extra} has no {split_name}")
            rename = {c: (f"{extra.name}:{c}" if (c in launch or c in update) else c) for c in extra_wanted}
            load_forecast_levels(fit_dirs[0], 0.1, launch, update, side_info=side_info, wanted=extra_wanted, rename=rename, point_index=point_index)

        for alpha, grid in level_grids[1:]:
            fit_dirs = list((grid / split_name).glob("fit-*"))
            if not fit_dirs:
                raise SystemExit(f"{grid} has no {split_name}")
            load_forecast_levels(fit_dirs[0], alpha, launch, update, side_info=side_info, point_index=point_index)
        if args.common_update_points:
            common = None
            for candidate in update_ids:
                points = {(rid, seq) for rid, seqs in update.get(candidate, {}).items() for seq in seqs}
                common = points if common is None else common & points
            common = common or set()
            for candidate in update_ids:
                for run_id in list(update.get(candidate, {})):
                    for seq in list(update[candidate][run_id]):
                        if (run_id, seq) not in common:
                            del update[candidate][run_id][seq]
                    if not update[candidate][run_id]:
                        del update[candidate][run_id]
        if args.sort_levels:
            reordered[split_name] = {}
            for candidate in sorted(set(launch) | set(update)):
                launch_counts = sort_levels(launch[candidate].values()) if candidate in launch else (0, 0)
                update_counts = sort_levels(lvs for seqs in update[candidate].values() for _, lvs in seqs.values()) if candidate in update else (0, 0)
                reordered[split_name][candidate] = {"launch": list(launch_counts), "update": list(update_counts)}
        checkpoint_counts[split_name] = {c: sum(len(seqs) for seqs in update.get(c, {}).values()) for c in update_ids}
        predictor_costs = build_predictor_costs(runs, launch, update, point_index, query_costs=query_costs, self_predict_costs=self_predict_costs) if args.charge_predictor else {}
        levels = sorted({lv for c in update for r in update[c] for _, lvs in update[c][r].values() for lv in lvs})
        run_ids = sorted(set(launch[launch_ids[0]]))
        totals = sorted(runs[r].total_tokens for r in run_ids)
        budgets = sorted({int(np.quantile(totals, q)) for q in quantiles})
        budgets_out = budgets
        results: dict[tuple, dict] = {}
        base_records = build_records(runs, run_ids, launch[launch_ids[0]], update[update_ids[0]])
        for budget in budgets:
            for policy in ("no_control", "fixed_cap", "oracle"):
                results[(policy, "", 0.5, budget)] = replay(runs, run_ids, budget=budget, policy=policy)
                cross_check(base_records, budget, results[(policy, "", 0.5, budget)], policy=policy, level=0.5)
            for candidate in launch_ids:
                for level in levels:
                    if level not in next(iter(launch[candidate].values())):
                        continue
                    res = replay(runs, run_ids, budget=budget, policy="admission", launch=launch[candidate], level=level, predictor_prelaunch=predictor_costs.get(candidate, {}).get("prelaunch"), predictor_launch=predictor_costs.get(candidate, {}).get("launch"))
                    results[("admission", candidate, level, budget)] = res
            for candidate in update_ids:
                for level in levels:
                    if not has_level(update[candidate], level):
                        continue
                    res = replay(runs, run_ids, budget=budget, policy="abort", update=update[candidate], level=level, predictor_prelaunch=predictor_costs.get(candidate, {}).get("prelaunch"), predictor_update=predictor_costs.get(candidate, {}).get("update"))
                    results[("abort", candidate, level, budget)] = res
            for level in levels:
                if level not in next(iter(launch[args.ours_launch].values())) or not has_level(update[args.ours_update], level):
                    continue
                results[("combined", "ours", level, budget)] = replay(runs, run_ids, budget=budget, policy="combined", launch=launch[args.ours_launch], update=update[args.ours_update], level=level, predictor_prelaunch=predictor_costs.get(args.ours_update, {}).get("prelaunch"), predictor_launch=predictor_costs.get(args.ours_launch, {}).get("launch"), predictor_update=predictor_costs.get(args.ours_update, {}).get("update"))
            cap_completed = results[("fixed_cap", "", 0.5, budget)]["summary"]["completed_rate"]
            budgeted = [(key, result) for key, result in results.items() if key[3] == budget and key[0] != "no_control"]
            hard_cap_audit.append({
                "split": split_name, "budget": budget, "policy_points": len(budgeted),
                "run_replays": sum(len(result["per_run"]) for _, result in budgeted),
                "over_budget_replays": sum(row["spent_tokens"] > budget for _, result in budgeted for row in result["per_run"].values()),
                "max_spent_tokens": max(row["spent_tokens"] for _, result in budgeted for row in result["per_run"].values()),
                "policy_points_above_cap_completion": sum(result["summary"]["completed_rate"] > cap_completed + 1e-12 for _, result in budgeted),
            })
            for (policy, _, _, row_budget), result in results.items():
                if row_budget == budget and policy != "no_control" and result["summary"]["completed_rate"] > cap_completed + 1e-12:
                    raise AssertionError(f"{policy} exceeds fixed-cap completion at budget {budget}")
        for (policy, candidate, level, budget), res in results.items():
            all_rows.append({"split": split_name, "policy": policy, "candidate": candidate, "level": level, **res["summary"]})

        cap_points = {b: results[("fixed_cap", "", 0.5, b)] for b in budgets}
        families = [("combined", "ours")] + [("abort", c) for c in update_ids] + [("admission", c) for c in launch_ids]
        fold_of_run = None
        if args.level_selection == "crossfit":
            fold_of_task = task_folds(next((args.grid / split_name).glob("fit-*")), split_name)
            fold_of_run = {r: fold_of_task[runs[r].task_id] for r in run_ids}
        for family_key in families:
            candidates_points = [(k, v) for k, v in results.items() if k[0] == family_key[0] and k[1] == family_key[1]]
            for budget, cap in cap_points.items():
                if fold_of_run is not None:
                    best, chosen = crossfit_operating_point({k[2]: v for k, v in candidates_points if k[3] == budget}, cap["per_run"], fold_of_run, budget)
                    operating_point, levels_used = {"budget": budget, "level_by_fold": chosen}, sorted(set(chosen.values()))
                else:
                    eligible = [(k, v) for k, v in candidates_points if v["summary"]["completed_rate"] >= cap["summary"]["completed_rate"]]
                    if not eligible:
                        continue
                    best_key, best = min(eligible, key=lambda kv: kv[1]["summary"]["spent_tokens"])
                    operating_point, levels_used = {"level": best_key[2], "budget": best_key[3]}, [best_key[2]]
                n = best["summary"]["n_runs"]
                paired.append({"split": split_name, "family": f"{family_key[0]}:{family_key[1]}", "cap_budget": budget, "operating_point": operating_point, "levels": levels_used, "cap_completed": cap["summary"]["completed_rate"], "completed": best["summary"]["completed_rate"], "cap_tokens": cap["summary"]["spent_tokens"], "tokens": best["summary"]["spent_tokens"], "saving": 1 - best["summary"]["spent_tokens"] / cap["summary"]["spent_tokens"], "agent_tokens_per_run": best["summary"]["agent_tokens"] / n, "seconds_per_run": best["summary"]["seconds"] / n, "cap_seconds_per_run": cap["summary"]["seconds"] / n, "predictor_tokens_per_run": best["summary"]["predictor_tokens"] / n, "predictor_seconds_per_run": best["summary"]["predictor_seconds"] / n, "hard_cap_hits_per_run": best["summary"]["hard_cap_hits"] / n, "overhead_caused_loss_rate": best["summary"]["overhead_caused_loss_rate"], "tokens_paired": task_paired(best["per_run"], cap["per_run"], "spent_tokens", seed=SEED), "seconds_paired": task_paired(best["per_run"], cap["per_run"], "seconds", seed=SEED), "completed_paired": task_paired(best["per_run"], cap["per_run"], "completed", seed=SEED)})


    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in all_rows:
        grouped[(row["policy"], row["candidate"], row["level"], row["budget"])].append(row)
    mean_rows = []
    for key, rows in sorted(grouped.items(), key=lambda kv: (kv[0][3], kv[0][0], kv[0][1], kv[0][2])):
        mean_rows.append({"policy": key[0], "candidate": key[1], "level": key[2], "budget": key[3], "n_splits": len(rows), **{k: statistics.fmean(float(r[k]) for r in rows) for k in ("agent_tokens", "predictor_tokens", "predictor_seconds", "spent_tokens", "seconds", "completed_rate", "false_stop_rate", "overhead_caused_loss_rate", "refused", "aborted", "hard_cap_hits")}, "cost": None if any(r["cost"] is None for r in rows) else statistics.fmean(r["cost"] for r in rows)})
    none = {r["budget"]: r for r in mean_rows if r["policy"] == "no_control"}
    for row in mean_rows:
        base = none[row["budget"]]
        row["tokens_vs_none"] = row["spent_tokens"] / base["spent_tokens"]
        row["seconds_vs_none"] = row["seconds"] / base["seconds"] if base["seconds"] else None
    accuracy = forecast_accuracy(args.grid, args.main_table_json)
    matched_grouped: dict[tuple, list[dict]] = defaultdict(list)
    for p in paired:
        matched_grouped[(p["family"], p["cap_budget"])].append(p)
    matched = []
    for (family, cap_budget), items in sorted(matched_grouped.items()):
        def agg(key):
            return combine_task_paired([i[key] for i in items], seed=SEED)
        matched.append({"family": family, "forecast_mae": accuracy.get(family.split(":", 1)[-1]), "cap_budget": cap_budget, "n_splits": len(items), "cap_completed": statistics.fmean(i["cap_completed"] for i in items), "completed": statistics.fmean(i["completed"] for i in items), "cap_tokens": statistics.fmean(i["cap_tokens"] for i in items), "tokens": statistics.fmean(i["tokens"] for i in items), "saving": statistics.fmean(i["saving"] for i in items), **{k: statistics.fmean(i[k] for i in items) for k in ("agent_tokens_per_run", "seconds_per_run", "cap_seconds_per_run", "predictor_tokens_per_run", "predictor_seconds_per_run", "hard_cap_hits_per_run", "overhead_caused_loss_rate")}, "levels": sorted({lv for i in items for lv in i["levels"]}), "tokens_paired": agg("tokens_paired"), "seconds_paired": agg("seconds_paired"), "completed_paired": agg("completed_paired")})
    report = {
        "grid": str(args.grid), "runs": str(args.runs), "events": str(args.events), "level_grids": [(a, str(g)) for a, g in level_grids], "extra": [str(e) for e in args.extra],
        "hard_cap": True, "prediction_token_scope": "provider-accounted query tokens; local proxy-model input tokens are uncharged",
        "common_update_points": args.common_update_points, "charge_predictor": args.charge_predictor, "level_selection": args.level_selection,
        "sort_levels": args.sort_levels, "reordered_forecasts": reordered,
        "splits": split_names, "budgets": budgets_out, "success_definition": "terminal_status == completed",
        "checkpoint_counts": checkpoint_counts, "hard_cap_audit": hard_cap_audit,
        "mean_rows": mean_rows, "matched_completion": matched, "paired_per_split": paired,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "budget_control.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (args.output / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
    print((args.output / "summary.md").read_text(encoding="utf-8"))


def summary_markdown(report: dict) -> str:
    rows = report["mean_rows"]
    charge_predictor = report.get("charge_predictor", False)
    saving_label = "net saving (predictor charged)" if charge_predictor else "saving"
    lines = [f"# Budget-control replay ({report['grid']}; splits {', '.join(report['splits'])})", "",
             f"Success = {report['success_definition']}. Budgets = quantiles of the cohort's run totals: {report['budgets']}. Forecast levels from grids: {report['level_grids']}.", ""]
    lines += ["Every budgeted policy has a hard per-run token cap that includes recorded prediction-query tokens. The no-control row is an unbounded reference. A completed run must fit the cap including its prediction queries. Hard-cap stops between checkpoints use proportional agent wall time.", "",
              "Token accounting excludes locally processed proxy-model input tokens; configured forward times are included when predictor costs are charged. The fold-external level selection uses OOF forecasts whose training/calibration data can include the evaluated fold, so its tuning boundary is not fully isolated. Held-out completion can fall below the fixed-cap rate; compare the completion columns before interpreting a saving.", ""]
    lines += [f"Source logs: `{report['runs']}` and `{report['events']}`. Forecasts and split assignments come from the supplied grids and candidate exports. When predictor costs are charged, recorded pre-execution estimates are charged before agent execution; a prediction query cut by the hard cap has proportionally estimated query time.", ""]
    audit = report["hard_cap_audit"]
    lines += [f"Budget invariant audit: {sum(a['run_replays'] for a in audit):,} budgeted run replays, {sum(a['over_budget_replays'] for a in audit)} over-budget charges, {sum(a['policy_points_above_cap_completion'] for a in audit)} policy points exceeding fixed-cap completion.", ""]
    if report.get("extra"):
        lines.append(f"Extra candidate exports: {report['extra']}. Common Task-update checkpoints only: {report['common_update_points']}. Predictor cost charged: {charge_predictor}.")
        lines.append("")
    if report.get("sort_levels"):
        changed = {f"{split}/{c}": f"{v['update'][0]}/{v['update'][1]} update, {v['launch'][0]}/{v['launch'][1]} launch" for split, by_c in report["reordered_forecasts"].items() for c, v in by_c.items() if v["update"][0] or v["launch"][0]}
        lines += [f"Level values sorted into monotone order before the policies read them; forecasts reordered: {changed or 'none'}.", ""]
    candidates_seen = sorted({c for counts in report["checkpoint_counts"].values() for c in counts})
    lines += ["## Task-update checkpoints actually used, per split", "",
              "| split | " + " | ".join(candidates_seen) + " |",
              "|---|" + "---:|" * len(candidates_seen)]
    for split_name, counts in report["checkpoint_counts"].items():
        lines.append(f"| {split_name} | " + " | ".join(str(counts.get(c, 0)) for c in candidates_seen) + " |")
    selection = {"crossfit": "the level each outer test fold uses is the cheapest one completing at least as many of the other folds' runs as the cap at that budget", "in_sample": "ours' cheapest operating point completing at least as many runs"}[report.get("level_selection", "in_sample")]
    lines += ["", "Paired intervals resample tasks after averaging each task's difference over the selected split seeds. They are conditional on those split seeds.", "",
              f"## Fold-selected spend and completion against the fixed cap ({selection})", "",
              f"| family | Task-update MAE | cap budget | cap completed | policy completed | cap tokens | policy tokens | {saving_label} | levels used | hard-cap hits/run | overhead loss rate | Δ tokens per run [95% CI] | Δ seconds per run [95% CI] | Δ completed per run [95% CI] |" + (" predictor tokens/run | predictor seconds/run |" if charge_predictor else ""),
              "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|" + ("---:|---:|" if charge_predictor else "")]
    for m in report["matched_completion"]:
        t, s, c = m["tokens_paired"], m["seconds_paired"], m["completed_paired"]
        mae = "—" if m.get("forecast_mae") is None else f"{m['forecast_mae']:,.0f}"
        row = f"| {m['family']} | {mae} | {m['cap_budget']:,} | {m['cap_completed']:.3f} | {m['completed']:.3f} | {m['cap_tokens']:,.0f} | {m['tokens']:,.0f} | {m['saving']:+.1%} | {m['levels']} | {m['hard_cap_hits_per_run']:.3f} | {m['overhead_caused_loss_rate']:.3f} | {t['delta']:+,.0f} [{t['ci_lower']:+,.0f}, {t['ci_upper']:+,.0f}] | {s['delta']:+.1f} [{s['ci_lower']:+.1f}, {s['ci_upper']:+.1f}] | {c['delta']:+.3f} [{c['ci_lower']:+.3f}, {c['ci_upper']:+.3f}] |"
        if charge_predictor:
            row += f" {m['predictor_tokens_per_run']:,.0f} | {m['predictor_seconds_per_run']:.3f} |"
        lines.append(row)
    lines += ["", "## Frontier points (seed mean): completed rate and spend by policy, level and budget", ""]
    for budget in report["budgets"]:
        lines += [f"### budget {budget:,}", "", "| policy | candidate | level | tokens vs none | wall vs none | completed | false-stop | overhead loss | hard-cap hits | refused | aborted |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in sorted((r for r in rows if r["budget"] == budget), key=lambda r: (-r["completed_rate"], r["tokens_vs_none"])):
            wall = "—" if r["seconds_vs_none"] is None else f"{r['seconds_vs_none']:.3f}"
            lines.append(f"| {r['policy']} | {r['candidate']} | {r['level']:.2f} | {r['tokens_vs_none']:.3f} | {wall} | {r['completed_rate']:.3f} | {r['false_stop_rate']:.3f} | {r['overhead_caused_loss_rate']:.3f} | {r['hard_cap_hits']:.1f} | {r['refused']:.1f} | {r['aborted']:.1f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
