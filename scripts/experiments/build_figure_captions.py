from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.experiments.build_figures import (
    RUNS,
    best_outside_ours,
    load_predictions,
    number,
    position_of,
    read_table,
)
from scripts.experiments.figure_style import family_of

OURS_AT = {"task_initialization": "input_output_scaled_interval", "call_start": "call_start_ours",
           "in_call_visible": "in_call_ours", "in_call_billed": "in_call_ours_clock",
           "task_update": "task_update_ours"}
SEEDS = "three split seeds (20260719, 20260720, 20260721)"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def f1() -> str:
    fit = RUNS / "mt-v18" / "split-20260719" / "fit-20260719"
    by: dict = {}
    for record in load_predictions(fit, {"task_update_ours", "empirical", "in_call_ours"}):
        by.setdefault((position_of(record), record["candidate_id"]), []).append(record)
    launch = {r["run_id"]: r["target_value"] for r in by[("task_initialization", "empirical")]}
    median = float(np.median(list(launch.values())))
    run = min(launch, key=lambda r: abs(launch[r] - median))
    inside = [r for r in by[("in_call_visible", "in_call_ours")] if r["run_id"] == run]
    calls = len({r["logical_call_id"] for r in inside})
    errors = [abs(r["prediction"] - r["target_value"]) for r in inside]
    prior = [r["upper"] for r in by[("in_call_visible", "empirical")] if r["run_id"] == run]
    rows, _ = read_table(RUNS / "mt-v18" / "table" / "summary.md")
    cohort = number(rows["in_call_visible"]["in_call_ours"][1])
    return (
        f"**One run, four moments.** A single v18 Task ({run.split('--')[1]}), split seed 20260719, "
        f"chosen as the run whose Task-launch target is nearest the cohort median "
        f"({launch[run]:,.0f} against a median of {median:,.0f} Tokens over {len(launch)} runs; no second "
        f"criterion was applied). (a) Tokens the run still needs after each call: the step line is the "
        f"realized remainder, the solid line the point forecast, the filled band its 90% split-conformal "
        f"interval at alpha 0.1, and the dashed pair the empirical-quantile prior at the same nominal "
        f"level. The forecast at k=0 is the Task-launch arm, so the panel shows the handoff from a "
        f"Task-only forecast to one that reads the Token ledger. (b) Every in-call checkpoint of the same "
        f"run in run order, {len(inside)} points across {calls} Calls, remaining visible output in UTF-8 "
        f"bytes; faint rules mark Call boundaries and the dashed line is the prior's upper bound, which "
        f"is flat at {max(prior):,.0f} bytes by construction. Narrower is better at equal coverage. This "
        f"run's in-call MAE is {np.mean(errors):.1f} bytes against the cohort's {cohort:.1f}, so it is "
        f"typical rather than favourable. Marks are one realization, not an average. "
        f"Source: `workspace/runs/mt-v18` and `workspace/runs/ti-v18`."
    )


def f2() -> str:


    from scripts.experiments.build_figures import POSITIONS

    fit = sorted((RUNS / "mt-v18").glob("split-*/fit-*"))[0]
    seen: dict[str, str] = {}
    targets: dict[str, list[float]] = {}
    for record in load_predictions(fit):
        position = position_of(record)
        seen.setdefault(position, record["candidate_id"])
        if record["candidate_id"] == seen[position]:
            targets.setdefault(position, []).append(float(record["target_value"]))
    ratios = {p: float(np.percentile(targets[p], 99) / np.median(targets[p]))
              for p in POSITIONS if p in targets}
    counts = {p: len(targets[p]) for p in ratios}
    report = read_json(RUNS / "mt-v19" / "repeat" / "repeat_run_coverage.json")["dispersion"]
    marginal = report["marginal_median"]["wape"]
    oracle = report["oracle_loo_sibling_median"]["wape"]
    return (
        f"**The quantity is wide, and repeating the Task does not narrow it.** (a) Distribution of each "
        f"position's target on v18, each divided by that position's own median so five different units "
        f"share one axis; log y, the white dot is the median, the thick bar the interquartile range, and "
        f"the figure above each violin is the 99th percentile over the median. That ratio runs "
        f"{min(ratios.values()):.1f}x to {max(ratios.values()):.1f}x, widest at In-call visible, over "
        f"{min(counts.values()):,} to {max(counts.values()):,} scored points per position. Every position "
        f"is forecasting a quantity that spans an order of magnitude inside its own cell. (b) Task-launch "
        f"WAPE on v19, lower is better: the marginal median reaches {marginal:.3f} and a leave-one-out "
        f"oracle handed the same Task's other three runs reaches {oracle:.3f}. The distance between them "
        f"is what any Task-only forecast could still win; everything below the oracle is dispersion "
        f"between repeats of one Task and is not available from the Task statement. Sources: "
        f"`workspace/runs/mt-v18` and `workspace/runs/mt-v19/repeat/repeat_run_coverage.json`."
    )


def f3() -> str:
    rows, ours = read_table(RUNS / "mt-v18" / "table" / "summary.md")
    counts = {p: len(rows[p]) for p in rows}
    margins = {}
    for position in rows:
        rival = best_outside_ours(rows[position], ours[position])
        if rival:
            mine = number(rows[position][ours[position]][1])
            margins[position] = 100 * (mine - rival[1]) / rival[1]
    wins = sum(1 for v in margins.values() if v < 0)
    return (
        f"**Every candidate, at every position.** MAE of each candidate on v18 as a percentage above the "
        f"reported recipe, which is the zero line; {SEEDS}, task-macro. One dot per candidate, "
        f"{min(counts.values())} to {max(counts.values())} per position, grouped by family and coloured by "
        f"it: published output-length predictors adapted to proxy representations, generic learners on the "
        f"same features, LLM-prompted protocols, and non-learned references. The horizontal bar marks the "
        f"best member of each family. Lower is better. The recipe has the lowest MAE at {wins} of the five "
        f"positions, and the spread inside the published family is wider than its distance to us at the "
        f"two in-call cells. Note the per-position y scales differ: the field is flat at Task launch and "
        f"spread at In-call visible. Source: `workspace/runs/mt-v18/table`."
    )


def f4() -> str:
    curve = read_json(RUNS / "mt-v18" / "prefix-curve" / "prefix_progress.json")
    bands = curve["bands"]
    report = read_json(RUNS / "ia-v18" / "ablation" / "ablation_summary.json")
    clock = report["in_call_billed"]["ic_billed_without_clock"]
    clock_pct = 100 * clock["delta"] / clock["reference_mae"]
    return (
        f"**Where the in-call gain comes from.** (a) MAE against how far the Call has already run, v18, "
        f"{len(bands)} equal-count bands of produced share, log y, lower is better. The dashed line is the "
        f"empirical prior: the remainder shrinks by construction as a Call proceeds, so a falling curve "
        f"alone proves nothing, and the prior is drawn to show what falling looks like with no model in "
        f"it. (b) What each source carries, measured by removing it and refitting: the dot is the mean "
        f"change in MAE over {SEEDS} and the bar spans the per-seed values. Right of zero means the model "
        f"was using the source. The stream clock is worth {clock_pct:.1f}% at the billed in-call target, "
        f"where reasoning Tokens are billed and invisible. Source: "
        f"`workspace/runs/mt-v18/prefix-curve` and `workspace/runs/ia-v18/ablation`."
    )


def f5() -> str:
    return (
        "**The gain under six cuts.** MAE of the recipe against the strongest candidate outside our own "
        "family at that position, as a percentage of that candidate; negative is better and the dotted "
        "rule is that candidate. The comparator excludes our own ablation arms, which the main table "
        "names as the strongest baseline at the two in-call cells, so these margins are smaller than the "
        "table's. The six cuts differ in what they hold out and in how large a field the comparator is "
        "drawn from: the three cohort rows compare against the full grid, while the transfer and length "
        "rows have only the reference and generic arms available in their artifacts. The last row swaps "
        "the estimator rather than the data, refitting our features with a sequence GRU, and is marked "
        "with a square. Cells never run are marked n/a rather than left blank. Sources: "
        "`workspace/runs/mt-v18`, `mt-v21`, `mt-v19`, `ct-v18-to-v21`, `le-v18` and "
        "`sub-learner_substitution`, all read at build time."
    )


def f6() -> str:
    return (
        "**Width at matched coverage, reservation at matched risk.** Split-conformal intervals at alpha "
        f"0.1, 0.2 and 0.5, {SEEDS}; down is better in both rows. Top row: normalized interval width "
        "against realized coverage, with the dotted rule at the nominal 0.90. Bottom row: extra Tokens "
        "reserved against the realized exceedance rate, with the dotted rule at the 0.10 target risk. "
        "Every conformal arm over-covers at the in-call positions, which is why width is read at a matched "
        "coverage of 0.90 rather than at the nominal level. The three columns are the positions whose "
        "frontier artifacts exist; Task launch is omitted because the reported arm is not in them, and the "
        "published heads run their own non-conformal intervals and are not in them either. Sources: "
        "`workspace/runs/mt-v18/frontier-cs-ic` and `workspace/runs/mt-v19/frontier`."
    )


def f7() -> str:
    v18 = read_json(RUNS / "mt-v18" / "budget" / "budget_control.json")
    budgets = sorted({row["budget"] for row in v18.get("mean_rows", []) if "budget" in row})
    span = f"{min(budgets):,.0f} to {max(budgets):,.0f}" if budgets else "eight levels"
    return (
        f"**A better forecast is a cheaper run.** Offline replay of the collected traces under a budget "
        f"policy that aborts a run when the forecast says it will exceed its budget; {len(budgets)} "
        f"budgets spanning {span} Tokens, {SEEDS}, success defined as terminal status completed. (a) "
        f"Completion rate against Tokens spent over the cohort at the best decision level for each "
        f"budget; up and left is better. (b) Median Token saving at matched completion against the "
        f"Task-update MAE of the forecast that drives the policy, which is the dose-response the claim "
        f"rests on. (c) Rate of runs stopped that would have completed, against completion rate; down and "
        f"right is better. Every saving here is a replay at matched completion status, not an online "
        f"intervention and not a resolve-rate comparison. Source: `workspace/runs/mt-v18/budget` and "
        f"`workspace/runs/mt-v19/budget`."
    )


def f8() -> str:
    overhead = read_json(RUNS / "overhead-idle" / "online_overhead.json")
    rows, _ = read_table(RUNS / "mt-v18" / "table" / "summary.md")
    published = [number(c[1]) for n, c in rows["in_call_visible"].items() if family_of(n) == "published"]
    return (
        f"**What one forecast costs.** Median wall clock on the online path per prediction point against "
        f"In-call visible MAE on v18; both lower is better and the x axis is logarithmic. The recipe's "
        f"cost is the feature build plus the predict call. A published head first runs the proxy model "
        f"over a 1,921-Token window before its own head runs, and the eleven heads are drawn as one column "
        f"at that shared cost, spanning {min(published):.1f} to {max(published):.1f} bytes of MAE. Asking "
        f"the model for its own remaining length costs a provider round trip. The published figures of "
        f"0.7 to 12.6 ms in those papers assume the serving model's hidden states are already in hand; "
        f"this axis charges the forward pass that a black-box deployment has to run to obtain them. "
        f"Source: `workspace/runs/overhead-idle/online_overhead.json` with `mt-v18/table`, "
        f"`rlp-v18` and `se-mini-v18`. Measured on {overhead.get('device', 'the development machine')}."
    )


def a1() -> str:
    slices = read_json(RUNS / "mt-v18" / "slices" / "prediction_slices.json")["positions"]
    ours = slices["task_update"]["task_update_ours"]["coverage_by_quintile"]
    prior = slices["task_update"]["empirical"]["coverage_by_quintile"]
    ours_span = (min(q["coverage"] for q in ours), max(q["coverage"] for q in ours))
    prior_span = (min(q["coverage"] for q in prior), max(q["coverage"] for q in prior))
    visible = slices["in_call_visible"]["in_call_ours"]["coverage_by_share"]
    return (
        f"**Coverage inside a run.** Realized coverage of each arm's 90% interval on v18 conditioned on "
        f"how far the Call or the run has progressed, {SEEDS} pooled, sample-weighted; the dotted rule is "
        f"the nominal level and closer to it is better. (a) In-call visible by produced-share band: every "
        f"conformal arm climbs from about {visible[0]['coverage']:.2f} in the first band to "
        f"{visible[-1]['coverage']:.2f} in the last, so the drift is a property of the position rather "
        f"than of any method, and the pooled coverage column of the main table averages over it. (b) Task "
        f"update by call-index quintile, where the arms do separate: the recipe holds "
        f"{ours_span[0]:.3f} to {ours_span[1]:.3f} across the run while the empirical prior runs "
        f"{prior_span[0]:.3f} to {prior_span[1]:.3f}, over-covering in the middle of a run and "
        f"under-covering at its end. Source: `workspace/runs/mt-v18/slices`, built from `mt-v18` and "
        f"`rw-v18` by `scripts/experiments/summarize_prediction_slices.py`."
    )


def a2() -> str:
    rows, ours = read_table(RUNS / "mt-v18" / "table" / "summary.md")
    cell = rows["in_call_visible"]
    mine = number(cell[ours["in_call_visible"]][3])
    generic = number(cell["lightgbm_generic"][3])
    trail = number(cell["trail_last"][3])
    return (
        f"**The whole error distribution.** Weighted empirical CDF of absolute error on v18, {SEEDS} "
        f"pooled, one panel per cell, log x; left is better. Each family is represented by its lowest-MAE "
        f"member, named inside the panel because the choice differs by position. Markers show the median "
        f"and the 90th percentile of each curve. The mean reported in the main table is one summary of "
        f"these curves: at In-call visible the 90th percentiles are {mine:.1f}, {generic:.1f} and "
        f"{trail:.1f} bytes for the recipe, the generic learner and trail_last, so the advantage is "
        f"present in the body of the distribution and not only in its tail. Source: "
        f"`workspace/runs/mt-v18/slices`."
    )


def a3() -> str:
    table = read_json(RUNS / "mt-v18" / "table" / "main_table.json")
    panel = table["panels"]["mt-v18"]["experiments"]
    out = []
    for position in ("call_start", "task_update"):
        cell = next(v for k, v in panel.items() if k.split("--")[0] == position)
        mine = cell["arms"][cell["ours"]]["metrics"]
        near = [b["metrics"]["spearman"] for n, b in cell["arms"].items()
                if n != cell["ours"] and abs(b["metrics"]["mae"] - mine["mae"]) / mine["mae"] < 0.07]
        if near:
            out.append(f"at {position.replace('_', ' ')} the arms within 7% of our MAE span Spearman rho "
                       f"{min(near):.3f} to {max(near):.3f} against our {mine['spearman']:.3f}")
    return (
        "**Error and ordering are different axes.** Spearman rho between the forecast and the realized "
        f"value, against MAE as a multiple of the best candidate in the panel; v18, {SEEDS}, log x. Left "
        "and up is better. Two positions are shown because those are the two a controller consumes. "
        + "; ".join(out) + ". Arms several times the best are drawn hollow at the right edge and counted "
        "there rather than compressing the cluster the panel is about. Colours and shapes are the "
        "families of Figure 3. Source: `workspace/runs/mt-v18/table/main_table.json`."
    )


def a4() -> str:
    transfer = read_json(RUNS / "ct-v18-to-v21" / "condition_transfer.json")
    shots = transfer.get("shots", [])
    return (
        f"**Reasoning low to reasoning off.** MAE of each arm fitted on v18 and scored on v21, as a "
        f"percentage above the same arm fitted in domain; lower is better and zero is the in-domain fit. "
        f"The x axis is an ordered factor: zero-shot, then {', '.join(str(s) for s in shots)} target-cohort "
        f"Tasks made available to the fit, then both cohorts pooled, then the in-domain fit itself. The two "
        f"positions that read the Token ledger lose the most zero-shot; the two that read the generated "
        f"text barely move, because the text of a reply looks the same whether or not the model reasoned "
        f"before writing it. Task launch is absent from this artifact. Source: "
        f"`workspace/runs/ct-v18-to-v21/condition_transfer.json`."
    )


def a5() -> str:
    prompted = read_json(RUNS / "rlp-v18" / "summary.json")
    return (
        f"**Asking the model for its own length.** (a) MAE of each prompted protocol before and after "
        f"post-hoc calibration fitted on the training split, log x, lower is better; the hollow diamond is "
        f"the raw reply and the arrow ends at the calibrated value. Each row is in its own unit, named on "
        f"the axis. The raw ask is off by an order of magnitude at every position and calibration removes "
        f"most of it, which makes the calibrated arm a scale-corrected reading of the model's ordering "
        f"rather than its stated number. (b) Realized coverage of the nominal 90% interval on the points "
        f"the two protocols share. v18, {prompted['n_prompts']} in-call prompts of which "
        f"{prompted['n_parsed']} parsed, at a mean {prompted['mean_elapsed_s']:.1f} s per estimate. The "
        f"recipe's Task-update MAE on the shared replay points differs from the main table's, which is "
        f"computed over a larger point set; the panel reports the shared-point value. Source: "
        f"`workspace/runs/rlp-v18`, `se-mini-v18` and `pe-v18-mini`."
    )


FIGURES = [
    ("f1_one_run", "One run, four moments", f1),
    ("f2_target_spread", "The spread of the target and the repeat-run floor", f2),
    ("f3_method_by_position", "Every candidate at every position", f3),
    ("f4_in_call_source", "Where the in-call gain comes from", f4),
    ("f5_gain_under_cuts", "The gain under six cuts", f5),
    ("f6_interval_range", "The interval's operating range", f6),
    ("f7_budget_control", "Budget control", f7),
    ("f8_cost_vs_error", "What one forecast costs", f8),
    ("a1_conditional_coverage", "Coverage inside a run", a1),
    ("a2_error_ecdf", "The whole error distribution", a2),
    ("a3_error_vs_ordering", "Error against ordering", a3),
    ("a4_transfer_regimes", "Condition-transfer regimes", a4),
    ("a5_llm_protocols", "The prompted protocols", a5),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "docs" / "figures")
    args = parser.parse_args()

    markdown = ["# Figure captions", "",
                "Generated by `scripts.experiments.build_figure_captions` from the same artifacts the",
                "figures plot, so a rebuilt figure and its caption cannot disagree. Rerun it after any",
                "grid changes.", ""]
    for name, title, builder in FIGURES:
        try:
            caption = builder()
        except (FileNotFoundError, KeyError, StopIteration, ValueError) as error:
            print(f"skipped {name}: {type(error).__name__} {error}", flush=True)
            continue
        markdown += [f"## {name}", "", f"*{title}*", "", caption, ""]
        print(f"wrote caption for {name} ({len(caption)} chars)", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "captions.md").write_text("\n".join(markdown) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.output / 'captions.md'}", flush=True)


if __name__ == "__main__":
    main()
