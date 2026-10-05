from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

from token_prediction.config import load_config
from token_prediction.harness import run_experiment_seed_grid
from token_prediction.pipeline import run_configured_experiments

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "experiments" / "four_position_baseline_grid_v1.toml"
SPLIT_SEEDS = (20260719, 20260720, 20260721)


def capture_dirs(parent: Path) -> list[Path]:
    dirs: list[Path] = []
    for run_dir in sorted(parent.iterdir()):
        state_path = run_dir / "state.json"
        if not state_path.is_file():
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("capture_status") not in {"complete", "validated"}:
            continue
        capture = run_dir / state["capture_path"]
        if (capture / "capture.json").is_file():
            dirs.append(capture)
    if not dirs:
        raise SystemExit(f"no completed capture dirs under {parent}")
    return dirs


def capture_run_id(capture: Path) -> str:
    try:
        document = json.loads((capture / "capture.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read capture manifest under {capture}: {exc}") from exc
    task = document.get("task") if isinstance(document, dict) else None
    run_id = task.get("run_id") if isinstance(task, dict) else None
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f"capture manifest under {capture} has no run_id")
    return run_id


def published_supervision_run_ids(path: Path) -> frozenset[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read published run roster {path}: {exc}") from exc
    selected: set[str] = set()
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            document = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid published run at line {line_number}: {exc}") from exc
        run_id = document.get("run_id") if isinstance(document, dict) else None
        eligible = document.get("supervision_eligible") if isinstance(document, dict) else None
        if not isinstance(run_id, str) or not run_id.strip() or not isinstance(eligible, bool):
            raise ValueError(f"invalid published run identity at line {line_number}")
        if run_id in seen:
            raise ValueError(f"duplicate published run_id {run_id!r}")
        seen.add(run_id)
        if eligible:
            selected.add(run_id)
    if not selected:
        raise ValueError("published run roster contains no supervision-eligible runs")
    return frozenset(selected)


def select_capture_dirs(
    parents: Iterable[Path],
    *,
    allowed_run_ids: frozenset[str] | None = None,
) -> tuple[Path, ...]:
    selected: list[Path] = []
    seen: set[str] = set()
    for parent in parents:
        found = capture_dirs(parent)
        print(f"{len(found)} capture dirs from {parent}", flush=True)
        for capture in found:
            run_id = capture_run_id(capture)
            if run_id in seen:
                raise ValueError(f"duplicate input capture run_id {run_id!r}")
            seen.add(run_id)
            if allowed_run_ids is None or run_id in allowed_run_ids:
                selected.append(capture)
    if allowed_run_ids is not None:
        selected_ids = {capture_run_id(capture) for capture in selected}
        missing = sorted(allowed_run_ids - selected_ids)
        if missing:
            raise ValueError(f"published run roster is missing captures: {missing}")
        print(
            f"selected {len(selected)} published supervision captures; "
            f"excluded {len(seen - allowed_run_ids)} other captures",
            flush=True,
        )
    return tuple(selected)


def main() -> None:
    argv = sys.argv[1:]
    config_path = CONFIG
    if "--config" in argv:
        index = argv.index("--config")
        config_path = Path(argv[index + 1])
        del argv[index : index + 2]
    run_roster_path = None
    if "--run-roster" in argv:
        index = argv.index("--run-roster")
        run_roster_path = Path(argv[index + 1])
        del argv[index : index + 2]
    args = [arg for arg in argv if arg != "--seed-grid"]
    seed_grid = "--seed-grid" in argv
    parents, output = [Path(arg) for arg in args[:-1]], Path(args[-1])
    allowed_run_ids = (
        None
        if run_roster_path is None
        else published_supervision_run_ids(run_roster_path)
    )
    inputs = select_capture_dirs(parents, allowed_run_ids=allowed_run_ids)
    config = load_config(config_path)
    if seed_grid:
        run_experiment_seed_grid(
            config,
            inputs,
            fit_seeds=(config.seed,),
            split_seeds=SPLIT_SEEDS,
            output_dir=output,
        )
    else:
        run_configured_experiments(config, inputs, output_dir=output)
    print("done", flush=True)


if __name__ == "__main__":
    main()
