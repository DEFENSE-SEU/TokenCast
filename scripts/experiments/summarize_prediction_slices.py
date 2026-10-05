from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BANDS = 5
QUANTILES = np.linspace(0.0, 1.0, 201)
SHARE_POSITION = "in_call_visible"
INDEX_POSITION = "task_update"


def position_of(record: dict) -> str:
    return record["experiment_id"].split("--")[0]


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, probabilities: np.ndarray) -> np.ndarray:

    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    if cumulative[-1] <= 0:
        return np.full(len(probabilities), float("nan"))
    positions = (cumulative - 0.5 * weights) / cumulative[-1]
    return np.interp(probabilities, positions, values)


def equal_count_bands(values: np.ndarray, bands: int) -> np.ndarray:

    order = np.argsort(values, kind="stable")
    assigned = np.empty(len(values), dtype=int)
    assigned[order] = np.minimum((np.arange(len(values)) * bands) // len(values), bands - 1)
    return assigned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grid", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    errors: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    share_rows: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    index_rows: dict[str, list[tuple[str, int, float, float]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)

    files = [(grid, path) for grid in args.grid
             for path in sorted(grid.glob("split-*/fit-*/predictions.jsonl"))]
    if not files:
        raise SystemExit(f"no predictions under {args.grid}")
    claimed: dict[tuple[str, str], Path] = {}


    context: dict[str, tuple[float | None, int]] = {}
    for grid, path in files:
        split = path.parents[1].name
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("label_status", "observed") != "observed":
                    continue
                position = position_of(record)
                candidate = record["candidate_id"]
                if claimed.setdefault((position, candidate), grid) != grid:
                    continue
                target = record["target_value"]
                weight = float(record.get("sample_weight") or 0.0)
                errors[(position, candidate)].append((abs(record["prediction"] - target), weight))
                counts[position] += 1
                covered = float(record["lower"] <= target <= record["upper"])
                if position not in (SHARE_POSITION, INDEX_POSITION):
                    continue
                point = record["point_id"]
                if "progress_known_amount" in record:
                    context.setdefault(point, (record["progress_known_amount"],
                                               record["cutoff_event_seq"]))
                produced, cutoff = context.get(point, (None, None))
                if position == SHARE_POSITION and produced is not None:
                    total = produced + target
                    if total > 0:
                        share_rows[candidate].append((produced / total, covered, weight))
                elif position == INDEX_POSITION and cutoff is not None:
                    index_rows[candidate].append((f"{split}:{record['run_id']}",
                                                  cutoff, covered, weight))
        print(f"read {path}", flush=True)

    report: dict = {"grids": [str(g) for g in args.grid], "files": len(files), "bands": BANDS,
                    "weighted_by": "sample_weight", "n_points": dict(counts),
                    "positions": defaultdict(dict)}

    for (position, candidate), rows in errors.items():
        values = np.asarray([r[0] for r in rows], dtype=float)
        weights = np.asarray([r[1] for r in rows], dtype=float)
        total = weights.sum()
        report["positions"][position][candidate] = {
            "n": int(values.size),
            "mae": float((values * weights).sum() / total) if total > 0 else float("nan"),
            "error_quantiles": [float(v) for v in weighted_quantiles(values, weights, QUANTILES)],
        }

    for candidate, rows in share_rows.items():
        shares = np.asarray([r[0] for r in rows], dtype=float)
        covered = np.asarray([r[1] for r in rows], dtype=float)
        weights = np.asarray([r[2] for r in rows], dtype=float)
        band = equal_count_bands(shares, BANDS)
        block = report["positions"][SHARE_POSITION].setdefault(candidate, {})
        entries = []
        for b in range(BANDS):
            mask = band == b
            total = weights[mask].sum()
            entries.append({"band": int(b), "share_median": float(np.median(shares[mask])),
                            "coverage": float((covered[mask] * weights[mask]).sum() / total) if total > 0 else float("nan"),
                            "n": int(mask.sum())})
        block["coverage_by_share"] = entries

    for candidate, rows in index_rows.items():

        per_run: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
        for run, cutoff, covered, weight in rows:
            per_run[run].append((cutoff, covered, weight))
        hits: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for entries in per_run.values():
            entries.sort()
            n = len(entries)
            for rank, (_, covered, weight) in enumerate(entries):
                hits[min(rank * BANDS // n, BANDS - 1)].append((covered, weight))
        block = report["positions"][INDEX_POSITION].setdefault(candidate, {})
        quintiles = []
        for q in sorted(hits):
            total = sum(w for _, w in hits[q])
            quintiles.append({"quintile": q,
                              "coverage": sum(c * w for c, w in hits[q]) / total if total > 0 else float("nan"),
                              "n": len(hits[q])})
        block["coverage_by_quintile"] = quintiles

    report["positions"] = dict(report["positions"])
    args.output.mkdir(parents=True, exist_ok=True)
    target_path = args.output / "prediction_slices.json"
    target_path.write_text(json.dumps(report), encoding="utf-8", newline="\n")
    print(f"wrote {target_path}", flush=True)
    for position, block in report["positions"].items():
        print(f"  {position}: {len(block)} candidates, {counts[position]:,} points", flush=True)
    for grid in args.grid:
        taken = sorted({c for (_, c), g in claimed.items() if g == grid})
        print(f"  from {grid.name}: {len(taken)} candidate ids", flush=True)


if __name__ == "__main__":
    main()
