from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

from token_prediction.features.text_embedding import hashed_text_embedding

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "workspace/experiments/feedback_micro_20260912"
CALLS = ROOT / "workspace/experiments/reference_calls/calls.jsonl"
ACTIONS = ROOT / "workspace/experiments/call_action_features_v2/features.jsonl"
META = ROOT / "workspace/external/self_prediction_cohort/swe_bench_verified_test.parquet"
SEED, TASKS, CUTOFF, FOLDS = 20260912, 120, 10, 3
PARAMS = dict(objective="regression_l1", n_estimators=120, num_leaves=7,
              learning_rate=0.05, min_child_samples=15, reg_lambda=1.0,
              n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True,
              random_state=SEED)
ARMS = {"history": ("history",), "actions": ("history", "actions"),
        "feedback": ("history", "actions", "feedback"),
        "plan": ("history", "actions", "feedback", "plan"),
        "oracle_N": ("history", "actions", "feedback", "plan", "oracle")}


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rank(task, purpose):
    return hashlib.sha256(f"{SEED}:{purpose}:{task}".encode()).hexdigest()


def stats(values, name):
    values = np.asarray([np.nan if v is None else float(v) for v in values])

    return {f"{name}_{k}": float(v) for k, v in {
        "last": values[-1], "sum": values.sum(), "mean3": values[-3:].mean(),
        "change": values[-1] - values[-2], "std3": values[-3:].std(),
    }.items()}


def point(record, action, metadata):
    calls, summaries = record["calls"][:CUTOFF], action["calls"][:CUTOFF]
    assert len(calls) == len(summaries) == CUTOFF
    assert all(c["call_id"] == s["call_id"] for c, s in zip(calls, summaries))
    text, repo = metadata["problem_statement"], metadata["repo"]
    history = {f"task_hash_{i}": v for i, v in enumerate(hashed_text_embedding(text, dim=32))}
    repo_bucket = int(hashlib.sha256(repo.encode()).hexdigest()[:8], 16) % 16
    history.update({f"repo_{i}": float(i == repo_bucket) for i in range(16)})
    history.update(task_chars=len(text), task_lines=len(text.splitlines()),
                   first_request_chars=calls[0]["request_chars"], completed_calls=CUTOFF)
    for key in ("input_tokens", "output_tokens", "total_tokens", "request_chars",
                "request_messages", "tool_call_count"):
        history.update(stats([c.get(key) for c in calls], key))
    history["last_finish_stop"] = float(calls[-1]["finish_reason"] == "stop")
    history["last_finish_length"] = float(calls[-1]["finish_reason"] == "length")
    actions = {"summary_available_rate": np.mean([bool(s["available"]) for s in summaries])}
    for kind in ("read", "edit", "run", "task_tracking", "finish", "think"):
        values = [(s["action_counts"] or {}).get(kind, 0) if s["available"] else None
                  for s in summaries]
        actions.update(stats(values, f"action_{kind}"))
    for key in ("test_run_count", "action_argument_chars", "tool_observation_chars"):
        actions.update(stats([s.get(key) for s in summaries], key))
    feedback = {}
    feedback.update(stats([float(c["error"]) for c in calls], "call_error"))
    feedback.update(stats([(c.get("tools") or {}).get("failed") for c in calls], "tool_failed"))
    seen, repeats = set(), []
    for summary in summaries:
        targets = summary.get("target_hashes")
        repeats.append(None if targets is None else sum(t in seen for t in targets))
        seen.update(targets or [])
    feedback.update(stats(repeats, "repeated_read_edit_target"))
    plans = [(i, s["task_plan"]) for i, s in enumerate(summaries) if s.get("task_plan") is not None]
    plan = {"plan_available": float(bool(plans)), "observed_plan_updates": len(plans)}
    fields = ("total_items", "todo_items", "in_progress_items", "done_items", "text_chars")
    for field in fields:
        plan[f"plan_{field}"] = plans[-1][1][field] if plans else np.nan
        plan[f"plan_change_{field}"] = plans[-1][1][field] - plans[-2][1][field] if len(plans) > 1 else np.nan
    plan["calls_since_observed_plan"] = CUTOFF - 1 - plans[-1][0] if plans else np.nan
    blocks = dict(history=history, actions=actions, feedback=feedback, plan=plan)
    future = record["calls"][CUTOFF:]
    return dict(task=record["task_id"], trajectory=record["trajectory_id"], blocks=blocks,
                target=sum(c["input_tokens"] + c["output_tokens"] for c in future),
                remaining_input=sum(c["input_tokens"] for c in future), remaining_calls=len(future))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "results.json").exists():
        raise FileExistsError("Results already exist; preserve this frozen pilot.")
    records = [json.loads(line) for line in CALLS.open(encoding="utf-8")]
    actions = {r["trajectory_id"]: r for r in map(json.loads, ACTIONS.open(encoding="utf-8"))}
    metadata = {r["instance_id"]: r for r in pq.read_table(
        META, columns=["instance_id", "repo", "problem_statement"]).to_pylist()}
    selected = sorted({r["task_id"] for r in records}, key=lambda t: rank(t, "sample"))[:TASKS]
    folds = {t: i % FOLDS for i, t in enumerate(sorted(selected, key=lambda t: rank(t, "fold")))}
    rows, selected_records = [], []
    for record in records:
        if record["task_id"] not in folds or len(record["calls"]) < CUTOFF:
            continue
        summary = actions[record["trajectory_id"]]
        assert summary["task_id"] == record["task_id"]
        assert len(record["calls"]) == len(summary["calls"])
        for call in record["calls"]:
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                assert isinstance(call[key], int) and call[key] >= 0
            assert call["input_tokens"] + call["output_tokens"] == call["total_tokens"]
        selected_records.append(record)
        rows.append(point(record, summary, metadata[record["task_id"]]))
    assert len({r["trajectory"] for r in rows}) == len(rows)
    assert set(r["task"] for r in rows) == set(selected)

    probe = selected_records[0]
    summary = actions[probe["trajectory_id"]]
    before = point(probe, summary, metadata[probe["task_id"]])
    mutated = copy.deepcopy(probe)
    mutated["calls"].append({"input_tokens": 12345, "output_tokens": 6789})
    mutated["terminal"] = "unseen-future-status"
    after = point(mutated, summary, metadata[probe["task_id"]])
    for block in before["blocks"]:
        np.testing.assert_equal(list(before["blocks"][block].values()), list(after["blocks"][block].values()))
    assert after["target"] == before["target"] + 19134
    assert after["remaining_calls"] == before["remaining_calls"] + 1
    for r in rows:
        r["blocks"]["oracle"] = {"FUTURE_remaining_calls_DIAGNOSTIC_ONLY": r["remaining_calls"]}
    feature_names = {a: [f"{b}.{n}" for b in blocks for n in rows[0]["blocks"][b]] for a, blocks in ARMS.items()}
    protocol = dict(seed=SEED, task_count=TASKS, cutoff=CUTOFF, folds=folds,
        checkpoint="after recorded call 10 and its associated tool feedback",
        target="sum of recorded future input_tokens + output_tokens; includes N=0",
        sampling="hash-selected development tasks; include runs reaching observed call 10",
        weighting="equal task weight, equal eligible run weight within task",
        caveat="recorded calls lack explicit physical-attempt/retry identities",
        evaluation="one frozen 3-fold task-grouped OOF; no tuning or early stopping",
        model=PARAMS, features=feature_names, input_sha256={str(p.relative_to(ROOT)): sha(p) for p in [CALLS, ACTIONS, META]},
        code_sha256=sha(Path(__file__)), bootstrap_draws=5000,
        primary_comparison="feedback versus actions; plan versus feedback is separately reported",
        supplementary="oracle N is future-information diagnostic; direct N prediction is a separate target")
    write_json(OUT / "protocol.json", protocol)
    counts = Counter(r["task"] for r in rows)
    weights = np.array([1 / counts[r["task"]] for r in rows])
    weights /= weights.mean()
    fold_array = np.array([folds[r["task"]] for r in rows])
    y = np.array([r["target"] for r in rows], dtype=float)
    n = np.array([r["remaining_calls"] for r in rows], dtype=float)
    predictions = {arm: np.zeros(len(rows)) for arm in ARMS}
    n_predictions = {arm: np.zeros(len(rows)) for arm in ("history", "plan")}
    for arm, blocks in ARMS.items():
        x = np.array([[v for block in blocks for v in r["blocks"][block].values()] for r in rows], dtype=float)
        assert x.shape[1] == len(feature_names[arm]) and not np.isinf(x).any()
        for fold in range(FOLDS):
            train, test = fold_array != fold, fold_array == fold
            assert not ({r["task"] for r, take in zip(rows, train) if take} &
                        {r["task"] for r, take in zip(rows, test) if take})
            model = lgb.LGBMRegressor(**PARAMS).fit(x[train], y[train], sample_weight=weights[train])
            predictions[arm][test] = np.maximum(0, model.predict(x[test]))
            if arm in n_predictions:
                n_model = lgb.LGBMRegressor(**PARAMS).fit(x[train], n[train], sample_weight=weights[train])
                n_predictions[arm][test] = np.maximum(0, n_model.predict(x[test]))
        print(f"completed {arm}", flush=True)
    task_list = sorted(counts)
    task_masks = [np.array([r["task"] == t for r in rows]) for t in task_list]
    errors = {a: np.array([np.abs(p[m] - y[m]).mean() for m in task_masks]) for a, p in predictions.items()}
    n_errors = {a: np.array([np.abs(p[m] - n[m]).mean() for m in task_masks]) for a, p in n_predictions.items()}
    draws = np.random.default_rng(SEED).integers(0, len(task_list), size=(5000, len(task_list)))

    def compare(err, baseline, candidate):
        delta = err[candidate] - err[baseline]
        ci = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
        return dict(delta_mae=float(delta.mean()), relative_change_pct=float(100 * delta.mean() / err[baseline].mean()),
                    descriptive_task_bootstrap_95_ci=ci.tolist(), task_win_share=float(np.mean(delta < 0)))

    comparisons = {f"{b}_to_{a}": compare(errors, b, a) for b, a in
                   [("history", "actions"), ("actions", "feedback"), ("feedback", "plan"),
                    ("history", "plan"), ("plan", "oracle_N")]}
    ambiguity = sum(any(v > 1 for v in Counter((c["input_tokens"], c["output_tokens"], c["tool_call_count"])
                    for c in r["calls"]).values()) for r in selected_records)
    audit = dict(tasks=len(task_list), runs=len(rows), points=len(rows),
        short_runs_excluded=sum(r["task_id"] in folds and len(r["calls"]) < CUTOFF for r in records),
        zero_remaining_points=int(np.sum(n == 0)), input_usage_share=float(sum(r["remaining_input"] for r in rows) / y.sum()),
        plan_available_points=sum(r["blocks"]["plan"]["plan_available"] == 1 for r in rows),
        points_with_missing_action_summary=int(sum(r["blocks"]["actions"]["summary_available_rate"] < 1 for r in rows)),
        points_with_observed_tool_failure=sum(r["blocks"]["feedback"]["tool_failed_sum"] > 0 for r in rows),
        points_with_observed_target_repeat=sum(r["blocks"]["feedback"]["repeated_read_edit_target_sum"] > 0 for r in rows),
        trajectories_with_duplicate_usage_toolcount_alignment_keys=ambiguity,
        prefix_suffix_mutation_check="passed", accounting_reconciliation="passed", task_fold_disjointness="passed",
        feature_counts={a: len(v) for a, v in feature_names.items()},
        fold_mae={a: [float(np.average(np.abs(p[fold_array == f] - y[fold_array == f]), weights=weights[fold_array == f])) for f in range(FOLDS)] for a, p in predictions.items()})
    result = dict(audit=audit, mae={a: float(e.mean()) for a, e in errors.items()}, comparisons=comparisons,
        remaining_call_mae={a: float(e.mean()) for a, e in n_errors.items()},
        remaining_call_comparison=compare(n_errors, "history", "plan"),
        duration_seconds=time.perf_counter() - start,
        limitations=["Development data previously used in other exploratory experiments; no new holdout inference.",
                    "One checkpoint, one model configuration, one task split, small sample.",
                    "Feedback uses failure/repetition proxies and declared plan counts; no raw feedback semantic encoder.",
                    "Existing action projection aligns response groups by usage and tool count; no new raw-archive timing audit.",
                    "Oracle uses future call counts, not deployable and not a strict finite-sample performance bound.",
                    "Bootstrap describes held-out task variation conditional on this OOF fit; no refitting within bootstrap."])
    write_json(OUT / "results.json", result)
    with (OUT / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps(dict(task=r["task"], trajectory=r["trajectory"], fold=int(fold_array[i]),
                target=r["target"], remaining_calls=r["remaining_calls"],
                predictions={a: float(p[i]) for a, p in predictions.items()},
                remaining_call_predictions={a: float(p[i]) for a, p in n_predictions.items()})) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
