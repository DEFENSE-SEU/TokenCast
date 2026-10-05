from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.experiments.run_cross_domain_generalization import DOMAINS, load_domains
from scripts.pilots.run_actor_scale_calibration_pilot import (
    FAMILIES,
    FOLDS,
    POSITIONS,
    SEED,
    _fit,
    _metrics,
    _predict,
    _task_weights,
    _weighted_median,
    adapted_candidates,
    new_only_candidates,
    select_by_task_cv,
)

ROOT = Path(__file__).resolve().parents[2]
SIZES = (3, 5, 10, 20)


def run(output_dir: Path) -> dict:
    domains = load_domains()
    fold_of: dict[str, int] = {}
    for rows in domains.values():
        tasks = sorted({row.task_id for row in rows})
        random.Random(SEED).shuffle(tasks)
        fold_of.update({task: i % FOLDS for i, task in enumerate(tasks)})

    def rows_of(domain_list, actor, folds):
        return [row for d in domain_list for row in domains[d] if row.actor == actor and fold_of[row.task_id] in folds]

    report: dict = {"sizes": list(SIZES), "positions": {}}
    for position in POSITIONS:
        ratios: dict[str, dict[str, float]] = {}
        for actor in FAMILIES:
            for target in DOMAINS:
                sources = [d for d in DOMAINS if d != target]
                source = _fit(rows_of(sources, actor, {0, 2, 3, 4}), rows_of(sources, actor, {1}), position)
                collected: dict[str, list[np.ndarray]] = defaultdict(list)
                test_rows = []
                for fold in range(FOLDS):
                    test = rows_of([target], actor, {fold})
                    pool = rows_of([target], actor, {f for f in range(FOLDS) if f != fold})
                    test_rows.extend(test)
                    zero_shot = _predict(source, test)
                    collected["zero_shot"].append(zero_shot)
                    truth = np.asarray([row.targets[position] for row in pool])
                    collected["history_median"].append(np.full(len(test), _weighted_median(truth, _task_weights(pool))))
                    stop_fold = (fold + 1) % FOLDS
                    fit_rows = [row for row in pool if fold_of[row.task_id] != stop_fold]
                    stop_rows = [row for row in pool if fold_of[row.task_id] == stop_fold]
                    own = _fit(fit_rows, stop_rows, position)
                    collected["new_only_all"].append(_predict(own, test))
                    adapter = _fit(fit_rows, stop_rows, position, extra=_predict(source, fit_rows),
                                   valid_extra=_predict(source, stop_rows))
                    collected["adapted_all"].append(_predict(adapter, test, extra=zero_shot))
                    tasks = sorted({row.task_id for row in pool})
                    random.Random(SEED + fold).shuffle(tasks)
                    for size in SIZES:
                        chosen = set(tasks[:size])
                        subset = [row for row in pool if row.task_id in chosen]
                        forecast, _ = select_by_task_cv(subset, test, position, adapted_candidates(source, position))
                        collected[f"adapted_selected_{size}"].append(forecast)
                        forecast, _ = select_by_task_cv(subset, test, position, new_only_candidates(position))
                        collected[f"new_only_selected_{size}"].append(forecast)
                maes = {arm: _metrics(test_rows, np.concatenate(parts), position)["mae"] for arm, parts in collected.items()}
                ratios[f"{actor}|{target}"] = {arm: mae / maes["history_median"] for arm, mae in maes.items()}
                print(f"{position} {actor} {target}: zero-shot {ratios[f'{actor}|{target}']['zero_shot']:.3f}", flush=True)
        arms = list(next(iter(ratios.values())))
        geometric = {arm: math.exp(statistics.fmean(math.log(r[arm]) for r in ratios.values())) for arm in arms}
        by_target = {
            target: {arm: math.exp(statistics.fmean(math.log(r[arm]) for key, r in ratios.items() if key.endswith(f"|{target}")))
                     for arm in arms}
            for target in DOMAINS
        }
        report["positions"][position] = {"pairs": ratios, "geometric_mean": geometric, "by_target": by_target}

    lines = ["# Unseen task types: MAE relative to the target domain's history median (geometric mean)", ""]
    for position, block in report["positions"].items():
        arms = list(block["geometric_mean"])
        lines += [f"## {position}", "", "| target | " + " | ".join(arms) + " |", "|---" * (len(arms) + 1) + "|"]
        for target, values in block["by_target"].items():
            lines.append(f"| {target} | " + " | ".join(f"{values[a]:.3f}" for a in arms) + " |")
        lines.append("| all pairs | " + " | ".join(f"{block['geometric_mean'][a]:.3f}" for a in arms) + " |")
        lines.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Unseen task types on the public rollout corpus: a predictor trained on two task'.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "workspace" / "experiments" / "task_type_adaptation")
    run(parser.parse_args().output_dir)
    print("done", flush=True)


if __name__ == "__main__":
    main()
