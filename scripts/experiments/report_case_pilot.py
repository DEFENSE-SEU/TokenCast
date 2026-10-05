from __future__ import annotations

import argparse
import json
from pathlib import Path

from token_prediction.case_study.self_prediction import parse_estimate


POINT_NAMES = {"task_start": "Task Start", "call_start": "Call Start",
               "in_call_update": "In-call Update", "task_update": "Task Update"}


def numeric_diagnostic(row, raw):

    estimate = row["self_prediction"].get("estimate")
    if estimate is not None:
        return {"estimate": estimate, "strict_format_valid": True}
    text = raw.get("content")
    if not isinstance(text, str):
        return None
    lines = text.strip().splitlines()
    if len(lines) < 3 or lines[0].lower() not in {"```json", "```"} or lines[-1] != "```":
        return None
    try:
        estimate = parse_estimate("\n".join(lines[1:-1]), prediction_point=row["prediction_point"])
    except ValueError:
        return None
    return {"estimate": estimate, "strict_format_valid": False,
            "extraction": "removed sole outer Markdown fence; numeric values unchanged"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--additional-ledger", type=Path, action="append", default=[],
                        help="Include prior attempts in usage accounting without replacing predictions")
    args = parser.parse_args()
    results = json.loads((args.comparison / "paired-results.json").read_text(encoding="utf-8"))
    ledger = json.loads((args.comparison / "baseline-ledger.json").read_text(encoding="utf-8"))
    raw_by_id = {row["request_id"]: row for row in ledger["requests"]}
    diagnostics = {row["point_id"]: numeric_diagnostic(row, raw_by_id.get(row["point_id"], {}))
                   for row in results}
    probes = json.loads(args.probe.read_text(encoding="utf-8")) if args.probe else []
    actor_tokens = 0
    calls = 0
    lines = ["# Diagnostics from measured execution trajectories", "",
             "This comparison uses measured execution costs for preselected tasks. Termination reasons and task-verifier outcomes are recorded separately for each trajectory.",
             "This run uses an experimental implementation of the latest Method; the results do not establish method superiority or coverage across models.", ""]
    for directory in args.run_dir:
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        model = run["condition"]["actor_model_id"]
        total = sum(attempt["usage"]["total_tokens"] for call in state["calls"]
                    for attempt in call["attempts"] if attempt.get("usage"))
        actor_tokens += total
        calls += sum(len(call["attempts"]) for call in state["calls"])
        oracle_report = (state.get("oracle") or {}).get("report", {})
        passed = oracle_report.get("all_requirements_pass")
        outcome = "all requirements passed" if passed is True else "some requirements failed" if passed is False else "task-verifier outcome unavailable"
        lines.extend([f"## {model}", "", f"Task: {run['task']['task_id']}. "
                      f"{len(state['calls'])} logical calls, consuming {total:,} measured tokens.",
                      f"Termination reason: {state['launches'][-1]['actor_terminal']['reason']}. Task-verifier outcome: {outcome}.", "",
                      "| Prediction point | Actual tokens | TokenCast pilot | Self-Prediction | Absolute percentage errors |",
                      "|---|---:|---:|---:|---:|"])
        for row in results:
            if row["actor_model"] != model:
                continue
            actual, ours = row["labels"]["actual_tokens"], row["ours"]["median"]
            diagnostic = diagnostics[row["point_id"]]
            baseline = diagnostic["estimate"]["predicted_total_tokens"] if diagnostic else None
            ours_error = abs(ours-actual)/max(actual, 1)*100
            baseline_text = "No valid prediction" if baseline is None else f"{baseline:,}"
            if diagnostic and not diagnostic["strict_format_valid"]:
                baseline_text += "*"
            errors = f"{ours_error:.1f}% / " + ("—" if baseline is None else
                     f"{abs(baseline-actual)/max(actual, 1)*100:.1f}%")
            lines.append(f"| {POINT_NAMES[row['prediction_point']]} | {actual:,} | {ours:,.0f} | "
                         f"{baseline_text} | {errors} |")
        lines.append("")
    accounted_requests = list(ledger["requests"])
    for path in args.additional_ledger:
        accounted_requests.extend(json.loads(path.read_text(encoding="utf-8"))["requests"])
    settled = [row for row in accounted_requests if row["status"] == "complete"]
    baseline_tokens = sum(row["usage"]["total_tokens"] for row in settled)
    probe_tokens = sum(row.get("usage", {}).get("total_tokens", 0) for row in probes)
    usage = dict(actor_requests=calls, actor_tokens=actor_tokens,
                 baseline_requests=len(accounted_requests), baseline_tokens=baseline_tokens,
                 unsettled_baseline_requests=len(accounted_requests)-len(settled),
                 probe_requests=len(probes), probe_tokens=probe_tokens,
                 measured_total_tokens=actor_tokens+baseline_tokens+probe_tokens,
                 monetary_charge_cny=None)
    unknown_reserved = sum(row["reserved_tokens"] for row in accounted_requests
                           if row["status"] != "complete")
    usage["uncertain_requests_reserved_tokens"] = unknown_reserved
    lines.extend(["## Usage accounting and interpretation limits", "",
                  f"Task execution consumed {actor_tokens:,} tokens; Self-Prediction added {baseline_tokens:,} tokens; "
                  f"API probes consumed {probe_tokens:,} tokens. Total recorded usage is {usage['measured_total_tokens']:,} tokens. "
                  "The bill for charges in CNY is unavailable.", "",
                  f"Complete usage was unavailable for {usage['unsettled_baseline_requests']} additional requests. A local budget reservation of "
                  f"{unknown_reserved:,} tokens is retained; this amount represents a reservation only, with actual consumption unconfirmed.", "",
                  "An asterisk indicates that the outer Markdown code fence was the response's only format violation. The table parses the enclosed JSON with numeric values unchanged; "
                  "strict format compliance remains recorded as failed. No values are imputed for interrupted responses or responses that fail numeric consistency checks.", "",
                  "Task Start predicts the whole task; both Call positions predict the full current call; Task Update predicts the remaining task. "
                  "These prediction points have different target scopes, so their values cannot be summed across rows.", "",
                  "Input and output counts come from provider usage records and include repeated context input. At task start, the baseline receives a static snapshot of the original seed. "
                  "The baseline performed no additional tool exploration in this run; missing initial dynamic tool descriptions remain unknown.", "",
                  "The predictor was trained on separate historical tasks; provenance.json beside the checkpoint records the sources and task partitions. "
                  "New tasks, models, and execution configurations introduce distribution shifts. "
                  "Intervals transferred from source-domain calibration have no validated target-domain coverage. These development tasks are insufficient for ranking model capabilities.", "",
                  "Original prompts, baseline responses, and usage records are saved in this directory; full execution trajectories are saved in the respective run directories."])
    (args.comparison / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (args.comparison / "usage-summary.json").write_text(json.dumps(usage, indent=2), encoding="utf-8")
    (args.comparison / "numeric-diagnostics.json").write_text(json.dumps(diagnostics, indent=2),
                                                              encoding="utf-8")
    tex = [r"\subsection{Trajectory case studies}",
           "We compare preselected development trajectories at fixed causal cutoffs. "
           "All costs include provider-accounted input and output tokens. "
           "Execution termination and task-verifier outcomes are recorded separately in the accompanying report.",
           r"\begin{table}[ht]", r"\centering", r"\small",
           r"\begin{tabular}{llrrr}", r"\hline",
           r"Actor & Position & Actual & TokenCast pilot & Self-Prediction \\", r"\hline"]
    for row in results:
        actor = "Haiku" if "claude" in row["actor_model"] else "DeepSeek Flash"
        diagnostic = diagnostics[row["point_id"]]
        base = diagnostic["estimate"]["predicted_total_tokens"] if diagnostic else None
        base_text = "--" if base is None else f"{base:,}"
        if diagnostic and not diagnostic["strict_format_valid"]:
            base_text += r"$^{*}$"
        tex.append(f"{actor} & {POINT_NAMES[row['prediction_point']]} & "
                   f"{row['labels']['actual_tokens']:,} & {row['ours']['median']:,.0f} & "
                   + base_text + r" \\")
    tex.extend([r"\hline", r"\end{tabular}",
                r"\caption{Measured development diagnostics. Task Start predicts task total; "
                r"Call Start and In-call Update predict full call cost; Task Update predicts remaining task cost. "
                r"The predictor was trained on separate historical tasks. These development cases do not "
                r"establish comparative model capability or general prediction superiority. "
                r"An asterisk marks numeric extraction from a sole Markdown code fence; strict format compliance failed.}",
                r"\label{tab:trajectory-case-studies}", r"\end{table}"])
    (args.comparison / "case_study_diagnostics.tex").write_text("\n".join(tex)+"\n", encoding="utf-8")
    print(json.dumps(usage, indent=2))


if __name__ == "__main__":
    main()
