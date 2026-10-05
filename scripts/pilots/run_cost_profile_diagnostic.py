from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts.pilots.run_cost_profile_ablation import _turn_output_tokens
from scripts.pilots.run_endpoint_evidence_pilot import (
    DEFAULT_SEED,
    _assign_split_seed,
    _build_points,
)
from scripts.pilots.run_horizon_factorization_pilot import (
    _remaining_calls,
    _trajectory_turn_counts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Model-free future-call cost profile diagnostic"
    )
    parser.add_argument(
        "--dialogues",
        type=Path,
        default=Path(
            "workspace/external/rollout_dialogues/sokoban.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-offset", type=int, default=40)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dialogues = args.dialogues.resolve()
    if not dialogues.is_file():
        raise FileNotFoundError(dialogues)

    from scripts.pilots.run_cross_actor_state_progress_pilot import _derive_archive

    archive, task_universe = _derive_archive(dialogues)
    archive_points, _built = _build_points(dialogues, archive)
    horizon = _remaining_calls(archive_points, _trajectory_turn_counts(dialogues))
    costs = _turn_output_tokens(dialogues)
    points = _assign_split_seed(archive_points, args.split_seed, task_universe)

    raw: dict[int, list[tuple[float, float]]] = {}
    normalised: dict[int, list[tuple[float, float]]] = {}
    for point in points:
        remaining = int(horizon[point.point_id])
        if remaining <= 0:
            continue
        sequence = costs[point.trajectory_id]
        future = [
            float(sequence[point.update_index + offset - 1])
            for offset in range(1, remaining + 1)
            if point.update_index + offset - 1 < len(sequence)
        ]
        if not future:
            continue
        scale = float(np.mean(future))
        for offset, value in enumerate(future, start=1):
            if offset > args.max_offset:
                break
            raw.setdefault(offset, []).append((value, point.sample_weight))
            if scale > 0.0:
                normalised.setdefault(offset, []).append(
                    (value / scale, point.sample_weight)
                )

    def _weighted_mean(pairs: list[tuple[float, float]]) -> float:
        values = np.asarray([value for value, _ in pairs], dtype=np.float64)
        weights = np.asarray([weight for _, weight in pairs], dtype=np.float64)
        return float(np.sum(values * weights) / np.sum(weights))

    rows: list[dict[str, float | int]] = []
    for offset in sorted(raw):
        rows.append(
            {
                "offset": offset,
                "survivors": len(raw[offset]),
                "raw_mean_cost": _weighted_mean(raw[offset]),
                "normalised_mean_cost": (
                    _weighted_mean(normalised[offset]) if offset in normalised else 0.0
                ),
            }
        )

    first = rows[0]["raw_mean_cost"] if rows else 0.0
    print(f"{'j':>4}{'survivors':>11}{'raw mu(j)':>12}{'/mu(1)':>9}{'normalised':>12}")
    for row in rows:
        ratio = row["raw_mean_cost"] / first if first else 0.0
        print(
            f"{row['offset']:>4}{row['survivors']:>11}"
            f"{row['raw_mean_cost']:>12.1f}{ratio:>9.3f}"
            f"{row['normalised_mean_cost']:>12.3f}"
        )


    normalised_values = [row["normalised_mean_cost"] for row in rows]
    summary = {
        "split_seed": args.split_seed,
        "max_offset": args.max_offset,
        "normalised_min": float(np.min(normalised_values)),
        "normalised_max": float(np.max(normalised_values)),
        "normalised_span": float(
            np.max(normalised_values) - np.min(normalised_values)
        ),
        "raw_min": float(np.min([row["raw_mean_cost"] for row in rows])),
        "raw_max": float(np.max([row["raw_mean_cost"] for row in rows])),
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "profile_shape.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"\nnormalised span {summary['normalised_span']:.3f} "
        f"(min {summary['normalised_min']:.3f}, max {summary['normalised_max']:.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
