from __future__ import annotations

import json
from pathlib import Path

from token_prediction.experiments.input_roster import build_dsh_input_roster


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "configs" / "experiments" / "task_balanced_retrieval_inputs_v1.json"
SOURCES = (
    {
        "source_key": "v18",
        "dataset_directory": "workspace/datasets/swebench-dual-model-v1/gpt-oauth-v18/main",
        "runs_root": (
            "workspace/datasets/swebench-dual-model-v1/"
            "recollection-gpt-oauth-v18/runs/runs"
        ),
    },
    {
        "source_key": "v19",
        "dataset_directory": "workspace/datasets/swebench-dual-model-v1/gpt-oauth-v19/main",
        "runs_root": (
            "workspace/datasets/swebench-dual-model-v1/"
            "recollection-gpt-oauth-v19/runs/runs"
        ),
    },
    {
        "source_key": "v21",
        "dataset_directory": "workspace/datasets/swebench-reasoning-off-v21/main",
        "runs_root": "workspace/runs/reasoning-off-v21-main/runs",
    },
)


def main() -> None:
    roster = build_dsh_input_roster(
        ROOT,
        roster_id="task-balanced-retrieval-inputs-v1",
        config_id="task-balanced-retrieval-ablation-v1",
        sources=SOURCES,
    )
    OUTPUT.write_text(
        json.dumps(roster, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
