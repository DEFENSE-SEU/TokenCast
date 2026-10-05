from __future__ import annotations

import json
import tomllib
from pathlib import Path

from scripts.experiments.predict_public_two_actor_task_update import _final_prefix_rows
from scripts.experiments.run_public_trajectory_ood_transfer import _load_ood_tasks
from scripts.experiments.score_public_two_actor_task_start import sha256
from scripts.experiments.run_public_trajectory_validation import (
    _load_selected_tasks,
    _partition_task_ids,
    _read_task_ids,
)
from token_prediction.config import project_root_for


def main() -> None:
    config_path = Path("configs/experiments/public_trajectory_ood_transfer_v1.toml").resolve(strict=True)
    output = Path("workspace/experiments/public_two_actor_task_update_holdout_20260923_v2").resolve(strict=True)
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((output / "models.json").read_text(encoding="utf-8"))
    root = project_root_for(config_path)
    opus_path = root / config["source"]["local_path"]
    partition = _partition_task_ids(
        _read_task_ids(opus_path),
        seed=int(config["protocol"]["split_seed"]),
        final_holdout_fraction=float(config["protocol"]["final_holdout_fraction"]),
        pilot_tasks=int(config["protocol"]["pilot_tasks"]),
    )
    selected = frozenset(partition.final_holdout_tasks)
    ood = config["ood_source"]
    _, gpt_tasks = _load_ood_tasks(
        root / ood["local_directory"], ood["files"], ood["sha256"], selected,
    )
    opus_tasks = _load_selected_tasks(opus_path, selected)
    count = 0
    with (output / "predictions.jsonl").open("r", encoding="utf-8") as frozen:
        for actor, tasks in (("opus", opus_tasks), ("gpt_oss", gpt_tasks)):
            for row in _final_prefix_rows(tasks, actor, checkpoint["actors"][actor]):
                encoded = json.dumps(row, ensure_ascii=False) + "\n"
                recorded = frozen.readline()
                if encoded != recorded:
                    raise ValueError(f"causal replay differs at row {count + 1}: {row['task_id']}/{actor}/{row['call_index']}")
                count += 1
        if frozen.readline():
            raise ValueError("frozen prediction file has extra rows")
    replay_hash = sha256(output / "predictions.jsonl")
    if replay_hash != manifest["predictions_sha256"] or count != manifest["prediction_count"]:
        raise ValueError("causal replay differs from frozen predictions")
    receipt = {
        "predictions_sha256": replay_hash,
        "model_checkpoint_sha256": manifest["model_checkpoint_sha256"],
        "prediction_rows": count,
        "used_future_suffix_labels": False,
        "replay_match": True,
    }
    (output / "replay_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
