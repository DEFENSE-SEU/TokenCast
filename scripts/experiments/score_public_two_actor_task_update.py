from __future__ import annotations

import argparse
import json
import math
import tomllib
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

from scripts.experiments.score_public_two_actor_task_start import (
    ACTORS,
    BOOTSTRAPS,
    CONFIG,
    EXPECTED_FINAL_ID_SHA256,
    ROOT,
    final_ids,
    id_hash,
    ids_from_parquet,
    read_json,
    sha256,
    source_paths,
    write_json,
)


DEFAULT_OUTPUT = ROOT / "workspace/experiments/public_two_actor_task_update_holdout_20260923_v2"
METHODS = ("candidate", "call_clock", "empirical")


def verify(output: Path) -> tuple[dict, list[dict], list[str], dict, list[Path]]:
    manifest = read_json(output / "manifest.json")
    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    opus, metadata, shards = source_paths(config)
    expected_sources = {
        "opus": config["source"]["sha256"],
        "metadata": config["task_metadata"]["sha256"],
        "gpt_oss": config["ood_source"]["sha256"],
    }
    actual_sources = {
        "opus": sha256(opus),
        "metadata": sha256(metadata),
        "gpt_oss": [sha256(path) for path in shards],
    }
    if actual_sources != expected_sources or manifest.get("source_sha256") != actual_sources:
        raise ValueError("pinned source files differ from configuration or manifest")
    if manifest.get("config_sha256") != sha256(CONFIG):
        raise ValueError("configuration changed after predictions")
    scripts = {
        "predictor_script_sha256": ROOT / "scripts/experiments/predict_public_two_actor_task_update.py",
        "scorer_script_sha256": Path(__file__),
        "development_script_sha256": ROOT / "scripts/experiments/develop_public_two_actor_task_update.py",
        "source_utils_script_sha256": ROOT / "scripts/experiments/score_public_two_actor_task_start.py",
        "validation_helper_sha256": ROOT / "scripts/experiments/run_public_trajectory_validation.py",
        "ood_helper_sha256": ROOT / "scripts/experiments/run_public_trajectory_ood_transfer.py",
    }
    if any(manifest.get(key) != sha256(path) for key, path in scripts.items()):
        raise ValueError("frozen prediction or scoring implementation changed")
    development_path = output / "development_report.json"
    if manifest.get("development_report_sha256") != sha256(development_path):
        raise ValueError("development report changed")
    development = read_json(development_path)
    for actor in ACTORS:
        metrics = development["actors"][actor]["metrics"]
        if metrics["causal_state_lightgbm"]["mae"] >= metrics["call_clock"]["mae"]:
            raise ValueError("candidate failed development gate for one actor")
    if (manifest.get("selected_candidate") != "causal_state_lightgbm"
            or manifest.get("selected_comparator") != "call_clock"):
        raise ValueError("candidate or comparator changed")
    if (manifest.get("target_unscored") is not True
            or manifest.get("all_prefixes_including_terminal") is not True
            or manifest.get("bootstrap_seed") != 20260902):
        raise ValueError("prediction/score protocol changed")
    if manifest.get("model_checkpoint_sha256") != sha256(output / "models.json"):
        raise ValueError("fitted model checkpoint changed")
    predictions_path = output / "predictions.jsonl"
    if manifest.get("predictions_sha256") != sha256(predictions_path):
        raise ValueError("frozen predictions changed")
    ids = ids_from_parquet(opus)
    if ids != ids_from_parquet(shards) or ids != ids_from_parquet(metadata):
        raise ValueError("actor or metadata task identities differ")
    holdout = final_ids(ids, config)
    if (len(holdout) != 146 or id_hash(holdout) != EXPECTED_FINAL_ID_SHA256
            or manifest.get("final_id_sha256") != EXPECTED_FINAL_ID_SHA256):
        raise ValueError("permanent final task roster changed")
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    by_pair: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        if set(row) != {"task_id", "actor", "call_index", "predictions"}:
            raise ValueError("prediction row schema is not exact and label-free")
        task_id, actor, k = row["task_id"], row["actor"], row["call_index"]
        if task_id not in holdout or actor not in ACTORS or type(k) is not int or k < 1:
            raise ValueError("prediction identity or completed-prefix index is invalid")
        if set(row["predictions"]) != set(METHODS):
            raise ValueError("prediction method roster changed")
        if any(type(value) is bool or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in row["predictions"].values()):
            raise ValueError("prediction value is invalid")
        by_pair.setdefault((task_id, actor), []).append(k)
    if len(by_pair) != len(holdout) * len(ACTORS) or manifest.get("prediction_count") != len(rows):
        raise ValueError("prediction task/actor roster is incomplete")
    for indices in by_pair.values():
        if sorted(indices) != list(range(1, len(indices) + 1)):
            raise ValueError("prediction must cover every completed prefix once")
    return manifest, rows, holdout, config, shards


def load_outputs(paths: Path | list[Path], selected: list[str]) -> dict[str, list[int]]:
    dataset = ds.dataset(str(paths) if isinstance(paths, Path) else [str(path) for path in paths], format="parquet")
    table = dataset.to_table(
        columns=["instance_id", "turns"],
        filter=ds.field("instance_id").isin(selected),
    )
    actual = {}
    for row in table.to_pylist():
        task_id = str(row["instance_id"])
        if task_id in actual:
            raise ValueError("duplicate task payload")
        turns = row["turns"]
        if not isinstance(turns, list) or not turns:
            raise ValueError("task has no turns")
        outputs = [json.loads(turn).get("osl") for turn in turns]
        if any(type(value) is not int or value <= 0 for value in outputs):
            raise ValueError("invalid provider output token label")
        actual[task_id] = outputs
    if set(actual) != set(selected):
        raise ValueError("final task payloads incomplete")
    return actual


def score(rows: list[dict], holdout: list[str], outputs: dict[str, dict[str, list[int]]], seed: int) -> tuple[dict, list[dict]]:
    predictions = {(row["task_id"], row["actor"], row["call_index"]): row["predictions"] for row in rows}
    per_task = []
    eligible = []
    for task_id in holdout:
        by_actor = {}
        for actor in ACTORS:
            values = outputs[actor][task_id]
            n = len(values)
            observed = [k for t, a, k in predictions if t == task_id and a == actor]
            if sorted(observed) != list(range(1, n + 1)):
                raise ValueError("predictions do not match every trajectory prefix")
            if n < 2:
                by_actor[actor] = {"eligible": False, "turns": n}
                continue
            suffix = np.cumsum(np.asarray(values[::-1], dtype=float))[::-1]
            targets = np.asarray([suffix[k] for k in range(1, n)], dtype=float)
            method_errors = {
                method: float(np.mean([
                    abs(predictions[(task_id, actor, k)][method] - targets[k - 1])
                    for k in range(1, n)
                ]))
                for method in METHODS
            }
            by_actor[actor] = {
                "eligible": True,
                "turns": n,
                "target_mean": float(np.mean(targets)),
                "mae": method_errors,
            }
        record = {"task_id": task_id, "by_actor": by_actor}
        per_task.append(record)
        if all(by_actor[actor]["eligible"] for actor in ACTORS):
            eligible.append(record)
    if not eligible:
        raise ValueError("no jointly eligible task families")
    target = np.asarray([[row["by_actor"][actor]["target_mean"] for actor in ACTORS] for row in eligible])
    errors = {
        method: np.asarray([[row["by_actor"][actor]["mae"][method] for actor in ACTORS] for row in eligible])
        for method in METHODS
    }
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(eligible), size=(BOOTSTRAPS, len(eligible)))
    sampled_target = target[draw].mean(axis=1)
    metric = {}
    sampled_macro = {}
    for method in METHODS:
        mae = errors[method].mean(axis=0)
        mean_target = target.mean(axis=0)
        metric[method] = {
            "macro_wape": float(np.mean(mae / mean_target)),
            "by_actor": {
                actor: {"mae": float(mae[i]), "mean_target": float(mean_target[i]), "wape": float(mae[i] / mean_target[i])}
                for i, actor in enumerate(ACTORS)
            },
        }
        sampled_macro[method] = (errors[method][draw].mean(axis=1) / sampled_target).mean(axis=1)
    comparisons = {}
    for reference in METHODS[1:]:
        macro_delta = sampled_macro["candidate"] - sampled_macro[reference]
        actor_delta = errors["candidate"][draw].mean(axis=1) - errors[reference][draw].mean(axis=1)
        comparisons[reference] = {
            "macro_wape_delta": metric["candidate"]["macro_wape"] - metric[reference]["macro_wape"],
            "macro_wape_delta_ci95": [float(v) for v in np.quantile(macro_delta, [0.025, 0.975])],
            "by_actor_mae_delta": {
                actor: {
                    "delta": metric["candidate"]["by_actor"][actor]["mae"] - metric[reference]["by_actor"][actor]["mae"],
                    "ci95": [float(v) for v in np.quantile(actor_delta[:, i], [0.025, 0.975])],
                }
                for i, actor in enumerate(ACTORS)
            },
        }
    result = {
        "planned_task_families": len(holdout),
        "paired_eligible_task_families": len(eligible),
        "unpaired_or_single_turn_families": [row["task_id"] for row in per_task if row not in eligible],
        "prediction_rows_including_terminal": len(rows),
        "metrics": metric,
        "candidate_minus": comparisons,
    }
    return result, per_task


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "score"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if (output / "score_started.json").exists() or (output / "score.json").exists():
        raise ValueError("one-shot scoring has already started")
    manifest, rows, holdout, config, shards = verify(output)
    if args.action == "verify":
        print(json.dumps({"prediction_rows": len(rows), "final_id_sha256": id_hash(holdout)}))
        return 0
    started = {
        "manifest_sha256": sha256(output / "manifest.json"),
        "predictions_sha256": manifest["predictions_sha256"],
        "first_future_suffix_label_computation_follows_this_receipt": True,
    }
    with (output / "score_started.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(started, indent=2, sort_keys=True) + "\n")
    opus, _, _ = source_paths(config)
    actual = {"opus": load_outputs(opus, holdout), "gpt_oss": load_outputs(shards, holdout)}
    result, per_task = score(rows, holdout, actual, int(manifest["bootstrap_seed"]))
    result.update({
        "manifest_sha256": started["manifest_sha256"],
        "predictions_sha256": manifest["predictions_sha256"],
        "target": "future provider-reported output-token sum after each nonterminal completed turn",
        "comparison": "candidate versus actor-specific call-clock; paired family bootstrap with 5000 draws",
        "paid_api_calls": 0,
    })
    with (output / "per_task_scored.jsonl").open("w", encoding="utf-8") as stream:
        for row in per_task:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(output / "score.json", result)
    print(json.dumps({"paired_families": result["paired_eligible_task_families"], "primary_delta": result["candidate_minus"]["call_clock"]["macro_wape_delta"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
