from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import tomllib
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "workspace/experiments/public_two_actor_task_start_holdout_20260923"
CONFIG = ROOT / "configs/experiments/public_trajectory_ood_transfer_v1.toml"
METHODS = ("candidate", "comparator", "empirical", "length")
ACTORS = ("opus", "gpt_oss")
BOOTSTRAPS = 5000
EXPECTED_FINAL_ID_SHA256 = "eb65acfa4dfc0ee3ab1c2d0e86f860bd369402b1b0472fe05eafa346071dcc22"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def source_paths(config: dict) -> tuple[Path, Path, list[Path]]:
    opus = ROOT / config["source"]["local_path"]
    metadata = ROOT / config["task_metadata"]["local_path"]
    ood = ROOT / config["ood_source"]["local_directory"]
    shards = [ood / name for name in config["ood_source"]["files"]]
    return opus, metadata, shards


def ids_from_parquet(paths: list[Path] | Path) -> set[str]:
    if isinstance(paths, Path):
        table = pq.read_table(paths, columns=["instance_id"])
    else:
        table = ds.dataset([str(path) for path in paths], format="parquet").to_table(columns=["instance_id"])
    ids = [str(value) for value in table.column("instance_id").to_pylist()]
    if len(ids) != 731 or len(set(ids)) != len(ids):
        raise ValueError("expected 731 unique task identities in each source")
    return set(ids)


def final_ids(all_ids: set[str], config: dict) -> list[str]:
    protocol = config["protocol"]
    ordered = sorted(all_ids)
    random.Random(int(protocol["split_seed"])).shuffle(ordered)
    count = round(len(ordered) * float(protocol["final_holdout_fraction"]))
    return sorted(ordered[:count])


def id_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def verify(output: Path) -> tuple[dict, list[dict], list[str], dict, list[Path]]:
    manifest_path = output / "manifest.json"
    predictions_path = output / "predictions.jsonl"
    manifest = read_json(manifest_path)
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
        raise ValueError("pinned source hashes changed")
    if manifest.get("config_sha256") != sha256(CONFIG):
        raise ValueError("configuration changed after prediction")
    if manifest.get("score_script_sha256") != sha256(Path(__file__)):
        raise ValueError("scoring implementation changed after prediction")
    development_script = ROOT / "scripts/experiments/develop_public_two_actor_task_start.py"
    if manifest.get("development_script_sha256") != sha256(development_script):
        raise ValueError("development implementation changed after prediction")
    report_path = output / "development_report.json"
    if manifest.get("development_report_sha256") != sha256(report_path):
        raise ValueError("development selection report changed")
    development = read_json(report_path)
    candidate = development["method"]["selected_arm"]
    references = ("actor_empirical_median", "actor_full_prompt_length", "shared_weight_0.00")
    comparator = min(references, key=lambda arm: (development["macro_wape"][arm], references.index(arm)))
    expected_aliases = {
        "candidate": candidate,
        "comparator": comparator,
        "empirical": "actor_empirical_median",
        "length": "actor_full_prompt_length",
    }
    if (manifest.get("selected_candidate") != candidate
            or manifest.get("selected_comparator") != comparator
            or manifest.get("prediction_aliases") != expected_aliases):
        raise ValueError("prediction aliases differ from development selection")
    if development["macro_wape"][candidate] >= development["macro_wape"][comparator]:
        raise ValueError("development candidate did not improve on the strong comparator")
    ids = ids_from_parquet(opus)
    if ids != ids_from_parquet(shards) or ids != ids_from_parquet(metadata):
        raise ValueError("source task identities differ")
    holdout = final_ids(ids, config)
    if (len(holdout) != 146 or id_hash(holdout) != EXPECTED_FINAL_ID_SHA256
            or manifest.get("final_id_sha256") != EXPECTED_FINAL_ID_SHA256):
        raise ValueError("final task roster changed")
    if manifest.get("prediction_count") != len(holdout) * len(ACTORS):
        raise ValueError("frozen prediction count differs")
    if manifest.get("predictions_sha256") != sha256(predictions_path):
        raise ValueError("prediction file changed")
    if manifest.get("target_unopened") is not True:
        raise ValueError("no pre-target receipt")
    if manifest.get("bootstrap_seed") != 20260902:
        raise ValueError("bootstrap seed differs")
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    expected_pairs = {(task_id, actor) for task_id in holdout for actor in ACTORS}
    observed_pairs = {(row["task_id"], row["actor"]) for row in rows}
    if len(rows) != len(expected_pairs) or observed_pairs != expected_pairs:
        raise ValueError("prediction roster is incomplete or duplicated")
    for row in rows:
        if set(row) != {"task_id", "actor", "predictions"} or set(row["predictions"]) != set(METHODS):
            raise ValueError("prediction row schema is not label-free and exact")
        if any(type(value) is bool or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in row["predictions"].values()):
            raise ValueError("prediction value is invalid")
    return manifest, rows, holdout, config, shards


def targets(paths: Path | list[Path], selected: list[str]) -> dict[str, int]:
    dataset = ds.dataset(str(paths) if isinstance(paths, Path) else [str(path) for path in paths], format="parquet")
    table = dataset.to_table(
        columns=["instance_id", "turns"],
        filter=ds.field("instance_id").isin(selected),
    )
    actual = {}
    for row in table.to_pylist():
        task_id = str(row["instance_id"])
        if task_id in actual:
            raise ValueError("duplicate final-holdout task payload")
        turns = row["turns"]
        if not isinstance(turns, list) or not turns:
            raise ValueError("missing final-holdout turns")
        values = [json.loads(turn).get("osl") for turn in turns]
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("invalid output-token target")
        actual[task_id] = sum(values)
    if set(actual) != set(selected):
        raise ValueError("final-holdout targets are incomplete")
    return actual


def score(rows: list[dict], holdout: list[str], truth: dict[str, dict[str, int]], seed: int) -> dict:
    by_pair = {(row["task_id"], row["actor"]): row["predictions"] for row in rows}
    y = np.asarray([[truth[actor][task_id] for actor in ACTORS] for task_id in holdout], dtype=float)
    error = {
        method: np.asarray([
            [abs(by_pair[(task_id, actor)][method] - truth[actor][task_id]) for actor in ACTORS]
            for task_id in holdout
        ], dtype=float)
        for method in METHODS
    }
    point = {}
    for method in METHODS:
        mae = error[method].mean(axis=0)
        mean_target = y.mean(axis=0)
        point[method] = {
            "macro_wape": float(np.mean(mae / mean_target)),
            "by_actor": {
                actor: {
                    "mae": float(mae[i]),
                    "mean_target": float(mean_target[i]),
                    "wape": float(mae[i] / mean_target[i]),
                }
                for i, actor in enumerate(ACTORS)
            },
        }
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(holdout), size=(BOOTSTRAPS, len(holdout)))
    draw_target = y[draw].mean(axis=1)
    draw_score = {
        method: (error[method][draw].mean(axis=1) / draw_target).mean(axis=1)
        for method in METHODS
    }
    comparison = {}
    for reference in METHODS[1:]:
        macro_delta = draw_score["candidate"] - draw_score[reference]
        actor_delta = error["candidate"][draw].mean(axis=1) - error[reference][draw].mean(axis=1)
        comparison[reference] = {
            "macro_wape_delta": point["candidate"]["macro_wape"] - point[reference]["macro_wape"],
            "macro_wape_delta_ci95": [float(v) for v in np.quantile(macro_delta, [0.025, 0.975])],
            "by_actor_mae_delta": {
                actor: {
                    "delta": point["candidate"]["by_actor"][actor]["mae"] - point[reference]["by_actor"][actor]["mae"],
                    "ci95": [float(v) for v in np.quantile(actor_delta[:, i], [0.025, 0.975])],
                }
                for i, actor in enumerate(ACTORS)
            },
        }
    return {"tasks": len(holdout), "actors": list(ACTORS), "metrics": point, "candidate_minus": comparison}


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
        print(json.dumps({"verified_predictions": len(rows), "final_id_sha256": id_hash(holdout)}))
        return 0
    start = {
        "manifest_sha256": sha256(output / "manifest.json"),
        "predictions_sha256": manifest["predictions_sha256"],
        "first_target_payload_access_follows_this_receipt": True,
    }
    with (output / "score_started.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(start, indent=2, sort_keys=True) + "\n")
    opus, _, _ = source_paths(config)
    truth = {
        "opus": targets(opus, holdout),
        "gpt_oss": targets(shards, holdout),
    }
    result = score(rows, holdout, truth, int(manifest["bootstrap_seed"]))
    result.update({
        "manifest_sha256": start["manifest_sha256"],
        "predictions_sha256": manifest["predictions_sha256"],
        "target": "sum of reported output tokens (osl) over all turns",
        "primary_comparison": "candidate minus comparator macro WAPE; 5000 paired task-family bootstrap draws",
        "paid_api_calls": 0,
    })
    write_json(output / "score.json", result)
    print(json.dumps({"tasks": len(holdout), "candidate_macro_wape": result["metrics"]["candidate"]["macro_wape"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
