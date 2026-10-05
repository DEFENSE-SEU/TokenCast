from __future__ import annotations

import argparse
import json
import math
import tomllib
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

from scripts.experiments.predict_public_pro_full_cost_retrospective import (
    ACTORS,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CONFIG,
    DEV_REPORT,
    FINAL_ID_SHA256,
    METHODS,
    OUTPUT,
    STRUCTURED_REPORT,
    _hash_ids,
    _verify_fit,
    _write_json_new,
)
from scripts.experiments.run_public_trajectory_validation import (
    _partition_task_ids,
    _read_task_ids,
    _sha256,
)
from token_prediction.config import project_root_for


def _source_paths(config_path: Path) -> tuple[Path, list[Path], dict]:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    root = project_root_for(config_path)
    opus = root / config["source"]["local_path"]
    ood = config["ood_source"]
    gpt = [root / ood["local_directory"] / name for name in ood["files"]]
    return opus, gpt, config


def verify(output: Path = OUTPUT, config_path: Path = CONFIG,
           report_path: Path = DEV_REPORT, structured_path: Path = STRUCTURED_REPORT) -> tuple[dict, dict, tuple[str, ...], Path, list[Path]]:
    output = output.resolve(strict=True)
    if any((output / name).exists() for name in ("score_started.json", "score.json", "per_task_scored.jsonl")):
        raise ValueError("one-shot scoring has already started")
    config_path = config_path.resolve(strict=True)
    receipt = _verify_fit(output, config_path, report_path.resolve(strict=True), structured_path.resolve(strict=True))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("fit_receipt_sha256") != _sha256(output / "fit_receipt.json"):
        raise ValueError("fit receipt changed")
    if manifest.get("model_sha256") != receipt["model_sha256"]:
        raise ValueError("model checkpoint changed")
    if manifest.get("predictor_sha256") != receipt["predictor_sha256"] or manifest.get("scorer_sha256") != receipt["scorer_sha256"]:
        raise ValueError("prediction or scoring implementation changed")
    if manifest.get("predictions_sha256") != _sha256(output / "predictions.jsonl"):
        raise ValueError("frozen predictions changed")
    if (manifest.get("final_id_sha256") != FINAL_ID_SHA256
            or manifest.get("paired_tasks") != 146
            or manifest.get("methods") != list(METHODS)
            or manifest.get("all_prefixes_including_terminal") is not True
            or manifest.get("target_unscored") is not True
            or manifest.get("retrospective_endpoint") is not True
            or manifest.get("bootstrap_seed") != BOOTSTRAP_SEED
            or manifest.get("bootstrap_draws") != BOOTSTRAP_DRAWS):
        raise ValueError("retrospective scoring protocol changed")
    opus, gpt, config = _source_paths(config_path)
    if _sha256(opus) != receipt["source_sha256"] or [_sha256(path) for path in gpt] != receipt["ood_sha256"]:
        raise ValueError("public source files changed")
    ids = _read_task_ids(opus)
    gpt_ids = [task_id for path in gpt for task_id in _read_task_ids(path)]
    if len(ids) != 731 or len(gpt_ids) != 731 or len(set(gpt_ids)) != 731 or set(ids) != set(gpt_ids):
        raise ValueError("paired source ID sets changed")
    protocol = config["protocol"]
    partition = _partition_task_ids(
        ids,
        seed=int(protocol["split_seed"]),
        final_holdout_fraction=float(protocol["final_holdout_fraction"]),
        pilot_tasks=int(protocol["pilot_tasks"]),
    )
    final = partition.final_holdout_tasks
    if len(final) != 146 or _hash_ids(final) != FINAL_ID_SHA256:
        raise ValueError("permanent Pro final IDs changed")
    predictions: dict[tuple[str, str, int], dict[str, float]] = {}
    by_pair: dict[tuple[str, str], list[int]] = {}
    with (output / "predictions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if set(row) != {"task_id", "actor", "call_index", "predictions"}:
                raise ValueError("prediction contains an extra field or omits a required field")
            task_id, actor, index = row["task_id"], row["actor"], row["call_index"]
            if task_id not in final or actor not in ACTORS or type(index) is not int or index < 1:
                raise ValueError("invalid prediction identity")
            values = row["predictions"]
            if set(values) != set(METHODS) or any(
                type(value) is bool or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                for value in values.values()
            ):
                raise ValueError("invalid prediction values")
            key = (task_id, actor, index)
            if key in predictions:
                raise ValueError("duplicate prediction")
            predictions[key] = values
            by_pair.setdefault((task_id, actor), []).append(index)
    if len(by_pair) != len(final) * len(ACTORS) or len(predictions) != manifest["prediction_count"]:
        raise ValueError("prediction pair coverage differs")
    if any(sorted(indices) != list(range(1, len(indices) + 1)) for indices in by_pair.values()):
        raise ValueError("predictions do not cover all completed prefixes")
    return manifest, predictions, final, opus, gpt


def _read_costs(paths: Path | list[Path], final: tuple[str, ...]) -> dict[str, list[int] | None]:
    dataset = ds.dataset(str(paths) if isinstance(paths, Path) else [str(path) for path in paths], format="parquet")
    table = dataset.to_table(
        columns=["instance_id", "turns"],
        filter=ds.field("instance_id").isin(list(final)),
    )
    result: dict[str, list[int] | None] = {}
    for row in table.to_pylist():
        task_id = str(row["instance_id"])
        if task_id in result:
            raise ValueError("duplicate final task payload")
        raw_turns = row["turns"]
        if not isinstance(raw_turns, list) or len(raw_turns) < 2:
            result[task_id] = None
            continue
        costs = []
        valid = True
        for index, raw in enumerate(raw_turns):
            try:
                turn = json.loads(raw)
                output, new_input, full_input = turn.get("osl"), turn.get("isl_new"), turn.get("isl")
                if (type(output) is not int or output <= 0 or type(new_input) is not int or new_input < 0
                        or (index > 0 and (type(full_input) is not int or full_input < new_input))):
                    valid = False
                    break
                if index > 0:
                    costs.append(full_input + output)
            except (TypeError, ValueError, AttributeError):
                valid = False
                break
        result[task_id] = costs if valid else None
    if set(result) != set(final):
        raise ValueError("final task payloads are incomplete")
    return result


def score_arrays(predictions: dict[tuple[str, str, int], dict[str, float]],
                 final: tuple[str, ...], costs: dict[str, dict[str, list[int] | None]],
                 seed: int = BOOTSTRAP_SEED, draws: int = BOOTSTRAP_DRAWS) -> tuple[dict, list[dict]]:

    per_task = []
    eligible = []
    for task_id in final:
        by_actor = {}
        for actor in ACTORS:
            future_costs = costs[actor][task_id]
            observed = sorted(k for t, a, k in predictions if t == task_id and a == actor)
            if future_costs is None:
                by_actor[actor] = {"eligible": False, "reason": "invalid_or_short_trajectory"}
                continue
            if observed != list(range(1, len(future_costs) + 2)):
                raise ValueError("prediction/trajectory prefix count differs")
            suffix = np.cumsum(np.asarray(future_costs[::-1], dtype=float))[::-1]
            targets = np.asarray(suffix, dtype=float)
            errors = {
                method: float(np.mean([
                    abs(predictions[(task_id, actor, k)][method] - targets[k - 1])
                    for k in range(1, len(future_costs) + 1)
                ])) for method in METHODS
            }
            by_actor[actor] = {
                "eligible": True,
                "turns": len(future_costs) + 1,
                "mean_target": float(targets.mean()),
                "mae": errors,
            }
        record = {"task_id": task_id, "by_actor": by_actor}
        per_task.append(record)
        if all(by_actor[actor]["eligible"] for actor in ACTORS):
            eligible.append(record)
    if not eligible:
        raise ValueError("no jointly eligible paired tasks")
    target = np.asarray([[row["by_actor"][actor]["mean_target"] for actor in ACTORS] for row in eligible])
    errors = {method: np.asarray([
        [row["by_actor"][actor]["mae"][method] for actor in ACTORS] for row in eligible
    ]) for method in METHODS}
    metric = {}
    for method in METHODS:
        metric[method] = {
            "macro_wape": float(np.mean(errors[method].sum(axis=0) / target.sum(axis=0))),
            "by_actor": {
                actor: {
                    "mae": float(errors[method][:, index].mean()),
                    "wape": float(errors[method][:, index].sum() / target[:, index].sum()),
                } for index, actor in enumerate(ACTORS)
            },
        }
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(eligible), size=(draws, len(eligible)))
    sampled_target = target[draw].sum(axis=1)
    comparisons = {}
    for reference in METHODS[1:]:
        delta = errors["candidate"][draw].sum(axis=1) - errors[reference][draw].sum(axis=1)
        macro_delta = np.mean(delta / sampled_target, axis=1)
        comparisons[reference] = {
            "macro_wape_delta": metric["candidate"]["macro_wape"] - metric[reference]["macro_wape"],
            "paired_task_bootstrap_ci95": [float(value) for value in np.quantile(macro_delta, [0.025, 0.975])],
        }
    result = {
        "planned_paired_tasks": len(final),
        "jointly_eligible_tasks": len(eligible),
        "excluded_task_ids": [row["task_id"] for row in per_task if row not in eligible],
        "eligibility_gate_at_most_5_percent_excluded": (len(final) - len(eligible)) / len(final) <= 0.05,
        "metrics": metric,
        "candidate_minus": comparisons,
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
        "retrospective_endpoint": True,
    }
    return result, per_task


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "score"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve(strict=True)
    manifest, predictions, final, opus, gpt = verify(output)
    if args.action == "verify":
        print(json.dumps({"paired_tasks": len(final), "predictions": len(predictions), "target_payloads_read": False}))
        return 0
    start = {
        "manifest_sha256": _sha256(output / "manifest.json"),
        "predictions_sha256": manifest["predictions_sha256"],
        "first_final_full_cost_label_read_follows_this_receipt": True,
    }
    _write_json_new(output / "score_started.json", start)
    costs = {"opus": _read_costs(opus, final), "gpt_oss": _read_costs(gpt, final)}
    result, per_task = score_arrays(predictions, final, costs)
    result.update({
        "target": "future sum of provider-reported input and output tokens after completed prefix",
        "first_turn_input_missing": True,
        "price_weighted_currency_claim": False,
        "natural_completion_claim": False,
        "manifest_sha256": start["manifest_sha256"],
        "predictions_sha256": start["predictions_sha256"],
        "paid_api_calls": 0,
    })
    with (output / "per_task_scored.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in per_task:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_json_new(output / "score.json", result)
    print(json.dumps({"eligible": result["jointly_eligible_tasks"], "macro_wape_delta_vs_call_clock": result["candidate_minus"]["call_clock"]["macro_wape_delta"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
