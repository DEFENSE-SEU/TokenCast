import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from token_prediction.collection.readers.dsh_session import (
    DshSessionReader,
    _locate_session_log,
    _read_session_log,
)
from token_prediction.case_study.evidence import _current_request
from token_prediction.contracts import EventType
from token_prediction.dataset.target.ledger import build_target_operational_ledger


def extract(run):
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    capture = run / state["capture_path"]
    trajectory = DshSessionReader().read(capture)
    _, native = _read_session_log(_locate_session_log(capture))
    calls = {c.logical_call_id: c for c in build_target_operational_ledger(trajectory.events).calls}
    rows = []
    for event in trajectory.events:
        if event.event_type != EventType.REQUEST_BUILT:
            continue
        request = _current_request(event, native)
        if request["source"] != "native_session_request_surface":
            continue
        byte_counts = {
            k: len(
                json.dumps(request[k], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )
            for k in ["system", "messages", "tools"]
        }
        call = calls[event.logical_call_id]

        actual = call.attempts[0].usage.input_tokens
        rows.append(
            dict(
                task_id=trajectory.task_id,
                run=str(run),
                call_index=len(rows),
                cutoff=event.event_seq,
                input_tokens=actual,
                system_bytes=event.payload.get("system_utf8_bytes", 0),
                full_surface_bytes=sum(byte_counts.values()),
                surface_parts=byte_counts,
                source=request["source"],
            )
        )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", type=Path, default=Path("workspace/runs/composable-case-pilot"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("cases", nargs="*", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((a.pilot / "provenance.json").read_text())
    partition = provenance["task_family_partition"]
    rows = []
    for number, source in enumerate(provenance["sources"]):
        rows.extend(extract(Path(source)))
        if number % 10 == 0:
            print("parsed", number + 1, flush=True)
    train = [r for r in rows if partition[r["task_id"]] == "train" and r["input_tokens"] > 0]
    validation = [
        r for r in rows if partition[r["task_id"]] == "validation" and r["input_tokens"] > 0
    ]
    models = {}
    scores = {}
    for feature in ["system_bytes", "full_surface_bytes"]:
        X = np.array([[1, r[feature]] for r in train], dtype=float)
        y = np.array([r["input_tokens"] for r in train])
        counts = {
            task: sum(r["task_id"] == task for r in train)
            for task in set(r["task_id"] for r in train)
        }
        weights = np.sqrt([1 / counts[r["task_id"]] for r in train])
        coeff, _ = nnls(X * weights[:, None], y * weights)
        models[feature] = coeff.tolist()
        for subset in ["first", "all"]:
            selected = [r for r in validation if subset == "all" or r["call_index"] == 0]
            errors = np.array(
                [
                    abs(max(0.0, coeff[0] + coeff[1] * r[feature]) - r["input_tokens"])
                    for r in selected
                ]
            )
            scores[feature + "/" + subset] = dict(
                n=len(selected),
                mae=float(errors.mean()),
                median_absolute_error=float(np.median(errors)),
            )
    median = float(np.median([r["input_tokens"] for r in train]))
    for subset in ["first", "all"]:
        selected = [r for r in validation if subset == "all" or r["call_index"] == 0]
        scores["train_median/" + subset] = dict(
            n=len(selected), mae=float(np.mean([abs(median - r["input_tokens"]) for r in selected]))
        )
    cases = []
    for run in a.cases:
        extracted = extract(run)
        if not extracted:
            continue
        r = extracted[0]
        for feature, coeff in models.items():
            r[feature + "_prediction"] = max(0.0, coeff[0] + coeff[1] * r[feature])
        cases.append(r)
    report = dict(
        protocol="Task-balanced nonnegative affine input-token proxy fitted on train only; fixed model; validation only, no calibration evaluation",
        split_counts={
            s: list(partition.values()).count(s) for s in ["train", "validation", "calibration"]
        },
        models=models,
        validation=scores,
        case_first_requests=cases,
        source_rows=rows,
        limitations=[
            "Reconstructed surface JSON is a causal byte proxy, not the provider exact wire serialization.",
            "GPT-trained token-per-byte mapping has no cross-provider accuracy guarantee.",
            "Zero observed input usage excluded from fitting and metrics.",
        ],
    )
    (a.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "source_rows"}, indent=2))


if __name__ == "__main__":
    main()
