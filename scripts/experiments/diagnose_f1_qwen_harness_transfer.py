from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from token_prediction.case_study.segments import query_from_row


ROOT = Path("workspace/runs/qwen38-segment-expansion-v2")
POSITIONS = ("task_start", "call_start", "in_call_update", "task_update")
BENCHMARKS = ("swebench", "searchr1", "mmlupro", "longbench")
HARNESS = ("dsh", "openhands")
ALPHA = 10.0
SEED = 20260923
BOOTSTRAPS = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected(batch: dict[str, np.ndarray]) -> np.ndarray:
    return np.flatnonzero(batch["whole_suffix"] & (batch["n"] > 0))


def paired_families(batch: dict[str, np.ndarray], indices: np.ndarray) -> set[str]:
    by_family: dict[str, set[str]] = defaultdict(set)
    by_run: dict[tuple[str, str], set[str]] = defaultdict(set)
    for i in indices:
        family, harness = str(batch["task_family_id"][i]), str(batch["harness"][i])
        by_family[family].add(harness)
        by_run[family, harness].add(str(batch["trajectory_id"][i]))
    if any(len(runs) != 1 for runs in by_run.values()):
        raise ValueError("expected one selected execution per family and harness")
    return {family for family, harnesses in by_family.items() if harnesses == set(HARNESS)}


def rows(batch: dict[str, np.ndarray], position: str, harness: str, families: set[str]) -> np.ndarray:
    indices = selected(batch)
    return indices[
        (batch["position"][indices] == position)
        & (batch["harness"][indices] == harness)
        & np.isin(batch["task_family_id"][indices], list(families))
    ]


def target(batch: dict[str, np.ndarray], position: str, indices: np.ndarray) -> np.ndarray:
    key = "label_call_cost" if position in {"call_start", "in_call_update"} else "label_cost"
    result = batch[key][indices].astype(np.float64)
    if np.any(result <= 0):
        raise ValueError("selected positive-horizon point has nonpositive provider-token target")
    return result


def features(batch: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    vectors = []
    for i in indices:
        query = query_from_row(batch, int(i))
        benchmark = str(batch["benchmark"][i])
        if benchmark not in BENCHMARKS:
            raise ValueError(f"unexpected benchmark {benchmark}")
        vectors.append(
            np.concatenate(
                (
                    query["task"][0],
                    query["state"][0],
                    query["observation"][0],
                    np.log1p(
                        [
                            float(query["spent"][0]),
                            float(query["confirmed_call"][0]),
                            float(query["completed_calls"][0]),
                            float(query["task_text_bytes"][0]),
                        ]
                    ),
                    [float(benchmark == name) for name in BENCHMARKS],
                )
            )
        )
    matrix = np.asarray(vectors, dtype=np.float64)
    if not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError("no finite causal features")
    return matrix


def equal_family_weights(batch: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    counts = Counter(map(str, batch["task_family_id"][indices]))
    return np.asarray([1 / counts[str(batch["task_family_id"][i])] for i in indices])


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values, sorted_weights = values[order], weights[order]
    return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), sorted_weights.sum() / 2)])


def historical_medians(batch: dict[str, np.ndarray], position: str, harness: str, families: set[str]) -> dict[str, float]:
    indices = rows(batch, position, harness, families)
    answer = {}
    for benchmark in BENCHMARKS:
        subset = indices[batch["benchmark"][indices] == benchmark]
        if len(subset):
            answer[benchmark] = weighted_median(
                target(batch, position, subset), equal_family_weights(batch, subset)
            )
    return answer


def predict(model, batch: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    log_prediction = model.predict(features(batch, indices))
    result = np.expm1(log_prediction)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite linear-probe forecast")
    return np.maximum(result, 0)


def family_errors(batch, indices, predictions) -> dict[str, float]:
    families: dict[str, list[float]] = defaultdict(list)
    for i, prediction, actual in zip(indices, predictions, target(batch, str(batch["position"][indices[0]]), indices)):
        families[str(batch["task_family_id"][i])].append(abs(float(prediction) - float(actual)))
    return {family: float(np.mean(errors)) for family, errors in families.items()}


def mae(errors: dict[str, float]) -> float:
    return float(np.mean(list(errors.values())))


def paired_ci(a: dict[str, float], b: dict[str, float], seed: int) -> list[float]:
    families = sorted(a)
    if families != sorted(b):
        raise ValueError("paired error sets differ")
    differences = np.asarray([a[family] - b[family] for family in families])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(families), size=(BOOTSTRAPS, len(families)))
    return [float(x) for x in np.quantile(differences[draws].mean(axis=1), [0.025, 0.975])]


def run(protocol_path: Path, output_dir: Path) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for name, expected in protocol["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"frozen source hash changed: {name}")
    if sha256(Path(__file__)) != protocol["script_sha256"]:
        raise ValueError("diagnostic script differs from frozen protocol")
    train = dict(np.load(ROOT / "train.npz", allow_pickle=False))
    validation = dict(np.load(ROOT / "validation.npz", allow_pickle=False))
    if set(train["partition"]) != {"train"} or set(validation["partition"]) != {"validation"}:
        raise ValueError("source split is mixed")
    train_families = paired_families(train, selected(train))
    val_families = paired_families(validation, selected(validation))
    if train_families & val_families:
        raise ValueError("training and validation task families overlap")
    outcomes = []
    for point_index, position in enumerate(POSITIONS):
        fitted = {}
        medians = {}
        for harness in HARNESS:
            indices = rows(train, position, harness, train_families)
            model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
            model.fit(
                features(train, indices),
                np.log1p(target(train, position, indices)),
                ridge__sample_weight=equal_family_weights(train, indices),
            )
            fitted[harness] = model
            medians[harness] = historical_medians(train, position, harness, train_families)
        domain_errors = {}
        for harness in HARNESS:
            indices = rows(validation, position, harness, val_families)
            true = target(validation, position, indices)
            baseline = np.asarray([medians[harness][str(validation["benchmark"][i])] for i in indices])
            history_errors = family_errors(validation, indices, baseline)
            own_errors = family_errors(validation, indices, predict(fitted[harness], validation, indices))
            domain_errors[harness] = {"own": own_errors, "history": history_errors}
            if harness == "openhands":
                transfer_errors = family_errors(validation, indices, predict(fitted["dsh"], validation, indices))
                domain_errors[harness]["dsh_transfer"] = transfer_errors
            outcomes.append({
                "position": position,
                "test_harness": harness,
                "validation_families": len(set(map(str, validation["task_family_id"][indices]))),
                "validation_points": len(indices),
                "history_median_mae": mae(history_errors),
                "within_harness_linear_probe_mae": mae(own_errors),
                "within_over_history": mae(own_errors) / mae(history_errors),
                **({
                    "dsh_to_openhands_linear_probe_mae": mae(transfer_errors),
                    "transfer_over_history": mae(transfer_errors) / mae(history_errors),
                    "transfer_minus_within_mae": mae(transfer_errors) - mae(own_errors),
                    "transfer_minus_within_task_bootstrap_95ci": paired_ci(transfer_errors, own_errors, SEED + point_index),
                } if harness == "openhands" else {}),
            })
    result = {
        "scope": "Qwen3.8 validation-only linear probe; no API calls; not PDF Table 8",
        "protocol_sha256": sha256(protocol_path),
        "paired_train_families": len(train_families),
        "paired_validation_families": len(val_families),
        "test_accessed": False,
        "outcomes": outcomes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.output)


if __name__ == "__main__":
    main()
