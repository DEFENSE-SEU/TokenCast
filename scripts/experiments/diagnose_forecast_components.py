import argparse
import json
from pathlib import Path
import numpy as np
import torch
from token_prediction.case_study.segments import from_trajectory, query_from_row
from token_prediction.collection.readers.dsh_session import (
    DshSessionReader,
    _locate_session_log,
    _read_session_log,
)
from token_prediction.estimators.composable_segment import ComposableSegmentModel

COMPONENTS = [
    "remaining_calls",
    "initial_input",
    "first_call_cost",
    "first_call_surplus",
    "future_cost",
    "future_mean_call_cost",
    "remaining_cost",
]


def diagnostics(model, batch, indices, samples):
    rows = []
    for begin in range(0, len(indices), 16):
        chosen = indices[begin : begin + 16]
        queries = [query_from_row(batch, int(i)) for i in chosen]
        query = {k: np.concatenate([q[k] for q in queries]) for k in queries[0]}
        q = model._query(query)
        with torch.no_grad(), torch.random.fork_rng():
            torch.manual_seed(20260919 + begin)
            total, first = model._compose_draw(
                q,
                q["allowance"].clamp(max=1),
                (q["allowance"] - 1).clamp_min(0),
                samples,
                differentiable=False,
            )

        def array(x):
            return x.detach().numpy()

        predicted = {
            "remaining_calls": array(total["n"]),
            "initial_input": array(first["initial_length"]),
            "first_call_cost": array(first["cost"]),
            "first_call_surplus": array(first["cost"] - first["initial_length"]),
            "future_cost": array(total["cost"] - first["cost"]),
            "remaining_cost": array(total["cost"]),
        }
        predicted["future_mean_call_cost"] = predicted["future_cost"] / np.maximum(
            predicted["remaining_calls"] - 1, 1
        )
        for j, i in enumerate(chosen):
            n = int(batch["n"][i])
            whole = float(batch["label_cost"][i])
            call = float(batch["label_call_cost"][i])
            length = float(batch["input_length"][i])
            actual = dict(
                remaining_calls=n,
                initial_input=length,
                first_call_cost=call,
                first_call_surplus=call - length,
                future_cost=whole - call,
                remaining_cost=whole,
                future_mean_call_cost=(whole - call) / max(n - 1, 1),
            )
            estimates = {key: float(np.median(predicted[key][j])) for key in COMPONENTS}
            rows.append(
                dict(
                    task_id=str(batch["task_id"][i]),
                    prediction_point=str(batch["position"][i]),
                    cutoff=int(batch["cutoff_event_seq"][i]),
                    actual=actual,
                    prediction=estimates,
                    error={k: estimates[k] - actual[k] for k in COMPONENTS},
                    probability_ends_after_first=float(np.mean(array(first["terminal"])[j] > 0.5)),
                )
            )
    return rows


def summarize(rows):
    result = {}
    for point in sorted(set(r["prediction_point"] for r in rows)):
        group = [r for r in rows if r["prediction_point"] == point]
        tasks = sorted(set(r["task_id"] for r in group))
        metrics = {}
        for component in COMPONENTS:
            metrics[component] = dict(
                task_macro_mae=float(
                    np.mean(
                        [
                            np.mean(
                                [abs(r["error"][component]) for r in group if r["task_id"] == t]
                            )
                            for t in tasks
                        ]
                    )
                ),
                task_macro_bias=float(
                    np.mean(
                        [
                            np.mean([r["error"][component] for r in group if r["task_id"] == t])
                            for t in tasks
                        ]
                    )
                ),
            )
        result[point] = dict(n_points=len(group), n_tasks=len(tasks), metrics=metrics)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("cases", nargs="*", type=Path)
    a = p.parse_args()
    model = ComposableSegmentModel.load(a.model)
    model.network.eval()
    data = dict(np.load(a.source / "segments.npz"))
    partition = json.loads((a.source / "provenance.json").read_text())["task_family_partition"]
    valid = (
        np.array([partition[str(f)] == "validation" for f in data["task_family_id"]])
        & (data["n"] > 0)
        & data["whole_suffix"]
    )
    indices = np.flatnonzero(valid)
    rows = diagnostics(model, data, indices, a.samples)
    cases = []
    for directory in a.cases:
        state = json.loads((directory / "state.json").read_text())
        capture = directory / state["capture_path"]
        trajectory = DshSessionReader().read(capture)
        _, native = _read_session_log(_locate_session_log(capture))
        run = json.loads((directory / "run.json").read_text())
        caps = run["cap_profile"]
        options = dict(
            request_surface_features=True,
            native_events=native,
            max_output_tokens=caps.get("max_attempt_output_tokens"),
            max_provider_accounted_tokens=caps.get("max_provider_accounted_tokens"),
            max_wall_seconds=caps.get("max_wall_seconds"),
            token_cap_enforcement=caps.get("token_cap_enforcement"),
        )
        if caps.get("max_logical_calls") is None:
            options["pilot_max_calls"] = model.config.max_horizon
        else:
            options["max_calls"] = caps["max_logical_calls"]
        batch = from_trajectory(trajectory, **options)
        selected = []
        for point in ["task_start", "call_start", "in_call_update", "task_update"]:
            ix = np.flatnonzero(
                (batch["position"] == point) & batch["whole_suffix"] & (batch["n"] > 0)
            )
            if len(ix):
                selected.append(int(ix[0]))
        cases.extend(
            dict(run=str(directory), **r)
            for r in diagnostics(model, batch, np.array(selected), a.samples)
        )
    first_rows = {}
    for row in rows:
        first_rows.setdefault((row["task_id"], row["prediction_point"]), row)
    report = dict(
        source_validation_first=summarize(list(first_rows.values())),
        model=str(a.model),
        samples=a.samples,
        source_validation=summarize(rows),
        source_validation_rows=rows,
        case_rows=cases,
        interpretation=[
            "First-call surplus equals output plus repeated retry input, not output alone when retries occur.",
            "Component predictions are marginal medians; medians need not add.",
            "Task-start and task-update first-call diagnostics concern the next call.",
            "Actual initial input is retrospective label only, never an unknown-L query feature.",
            "Validation tasks used for model selection; diagnostic scores are not final test claims.",
        ],
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "source_validation_rows"}, indent=2))


if __name__ == "__main__":
    main()
