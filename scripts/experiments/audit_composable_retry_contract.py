from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from token_prediction.collection.readers.dsh_session import DshSessionReader
from token_prediction.contracts import EventType, TokenUsage


ROOT = Path(__file__).resolve().parents[2]
PROVENANCE = ROOT / "workspace/runs/composable-multidomain-history-data-v1/provenance.json"
OUT = ROOT / "workspace/experiments/composable_retry_contract_20260923/audit.json"
SEGMENTS = ROOT / "workspace/runs/composable-multidomain-history-data-v1/segments.npz"
TERMINALS = {EventType.API_COMPLETED, EventType.API_FAILED}


def usage(event):
    item = TokenUsage.from_mapping(event.payload.get("usage"))
    if not item.is_complete or item.reported_total_matches is False:
        raise ValueError("incomplete or inconsistent attempt usage")
    return item


def audit_run(run_dir: Path):
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    if state.get("capture_status") not in {"complete", "validated"} or not state.get("supervision_eligible"):
        return None
    trajectory = DshSessionReader().read(run_dir / state["capture_path"])
    events = trajectory.events
    starts = [i for i, e in enumerate(events) if e.event_type == EventType.REQUEST_BUILT]
    calls = []
    for lo, hi in zip(starts, starts[1:] + [len(events)]):
        call_events = events[lo:hi]
        attempts = [e for e in call_events if e.event_type == EventType.API_ATTEMPT_STARTED]
        finished = {e.attempt_id: usage(e) for e in call_events if e.event_type in TERMINALS}
        if not attempts or set(finished) != {e.attempt_id for e in attempts}:
            raise ValueError(f"attempt boundary mismatch: {run_dir}")
        ordered = [finished[e.attempt_id] for e in attempts]
        q = int(ordered[0].input_tokens)
        retry_input = sum(int(e.input_tokens) for e in ordered[1:])
        output = sum(int(e.output_tokens) for e in ordered)
        total = sum(int(e.accounted_total_tokens) for e in ordered)
        if q + retry_input + output != total:
            raise ValueError(f"call cost mismatch: {run_dir}")
        calls.append((q, retry_input, output, total, len(ordered)))
    qs = [c[0] for c in calls]
    if qs:
        qs.append(qs[-1])
    for r in range(len(calls)):
        for s in range(r, len(calls)):
            n = s - r + 1
            g = qs[s + 1] - qs[r]
            b = sum(calls[j][1] + calls[j][2] + qs[j] - qs[r] for j in range(r, s + 1))
            if n * qs[r] + b != sum(c[3] for c in calls[r : s + 1]):
                raise ValueError(f"segment cost mismatch: {run_dir}")
            if s > r:
                cut = r + (s - r) // 2
                n_a = cut - r + 1
                g_a = qs[cut + 1] - qs[r]
                b_a = sum(calls[j][1] + calls[j][2] + qs[j] - qs[r] for j in range(r, cut + 1))
                n_b = s - cut
                g_b = qs[s + 1] - qs[cut + 1]
                b_b = sum(calls[j][1] + calls[j][2] + qs[j] - qs[cut + 1] for j in range(cut + 1, s + 1))
                if (n_a + n_b, g_a + g_b, b_a + b_b + n_b * g_a) != (n, g, b):
                    raise ValueError(f"composition mismatch: {run_dir}")
    return {
        "run_id": trajectory.run_id,
        "logical_calls": len(calls),
        "physical_attempts": sum(c[4] for c in calls),
        "retry_calls": sum(c[4] > 1 for c in calls),
        "retry_input_tokens": sum(c[1] for c in calls),
        "provider_tokens": sum(c[3] for c in calls),
        "segments_checked": len(calls) * (len(calls) + 1) // 2,
        "_calls": calls,
    }


def main():
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    rows = [audit_run(Path(p)) for p in provenance["sources"]]
    kept = [x for x in rows if x is not None]
    by_id = {x["run_id"]: x["_calls"] for x in kept}
    with np.load(SEGMENTS, allow_pickle=False) as archive:
        data = {key: archive[key] for key in
                ("run_id", "horizon", "n", "start_call", "input_length", "g", "b",
                 "label_cost", "label_call_cost")}
    checked_rows = 0
    for i, run_id in enumerate(data["run_id"]):
        calls = by_id.get(str(run_id))
        if calls is None or int(data["horizon"][i]) != 1 or int(data["n"][i]) != 1:
            continue
        j = int(data["start_call"][i])
        q, retry_input, output, total, _ = calls[j]
        next_q = calls[j + 1][0] if j + 1 < len(calls) else q
        observed = tuple(int(round(float(data[k][i]))) for k in
                         ("input_length", "g", "b", "label_cost", "label_call_cost"))
        expected = (q, next_q - q, retry_input + output, total, total)
        if observed != expected:
            raise ValueError(f"frozen segment row mismatch: {run_id} call {j}: {observed} vs {expected}")
        checked_rows += 1
    report = {
        "scope": "source SWE runs listed in composable-multidomain-history-data-v1 provenance",
        "sources_listed": len(rows),
        "eligible_runs_audited": len(kept),
        "skipped_ineligible": len(rows) - len(kept),
        "logical_calls": sum(x["logical_calls"] for x in kept),
        "physical_attempts": sum(x["physical_attempts"] for x in kept),
        "retry_calls": sum(x["retry_calls"] for x in kept),
        "retry_input_tokens": sum(x["retry_input_tokens"] for x in kept),
        "provider_tokens": sum(x["provider_tokens"] for x in kept),
        "segments_checked": sum(x["segments_checked"] for x in kept),
        "frozen_horizon_one_rows_checked": checked_rows,
        "all_call_and_segment_accounting_exact": True,
        "runs_with_retries": [x["run_id"] for x in kept if x["retry_calls"]],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
