from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.pilots.run_stream_timing_pilot import DEFAULT_RUNS, load_attempts

ROOT = Path(__file__).resolve().parents[2]
RATE_PERCENTILES = (25, 10, 5, 1)


def adjacent_rates(attempt) -> np.ndarray | None:


    times = np.asarray([c[0] for c in attempt.checkpoints], dtype=float)
    byte_counts = np.asarray([c[1] for c in attempt.checkpoints], dtype=float)
    if len(times) < 2 or attempt.visible_tokens <= 0:
        return None
    beta = byte_counts[-1] / attempt.visible_tokens
    seconds = np.diff(times) / 1000.0
    delta_bytes = np.diff(byte_counts)
    usable = (seconds > 0) & (delta_bytes > 0)
    return delta_bytes[usable] / (beta * seconds[usable]) if usable.any() else None


def fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:


    design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    return float(coefficients[0]), float(coefficients[1]), float(1 - residual.var() / y.var())


def run(runs_root: Path, output_dir: Path) -> None:
    attempts = [a for a in load_attempts(runs_root) if a.checkpoints and a.visible_tokens > 0]
    by_session = defaultdict(list)
    for attempt in attempts:
        by_session[attempt.session_id].append(attempt)
    for session in by_session.values():
        session.sort(key=lambda a: a.ordinal)

    billed = np.asarray([a.billed_output for a in attempts], dtype=float)
    hidden = np.asarray([a.billed_output - a.visible_tokens for a in attempts], dtype=float)
    duration = np.asarray([(a.end_ms - a.start_ms) / 1000.0 for a in attempts])
    ttfb = np.asarray([a.ttfb for a in attempts])

    startup_d, slope_d, r2_d = fit(billed, duration)
    startup_t, slope_t, r2_t = fit(hidden, ttfb)
    session_rates = []
    for session in by_session.values():
        rates = [r for r in (adjacent_rates(a) for a in session) if r is not None]
        if rates:
            session_rates.append(float(np.median(np.concatenate(rates))))

    bounds: dict[int, dict[str, float]] = {}
    for percentile in RATE_PERCENTILES:
        holds, ratios = [], []
        for session in by_session.values():
            seen_ttfb, seen_rates = [], []
            for attempt in session:
                if seen_ttfb and seen_rates:
                    startup = min(seen_ttfb)
                    rate = float(np.percentile(np.concatenate(seen_rates), percentile))
                    bound = rate * max(0.0, (attempt.end_ms - attempt.start_ms) / 1000.0 - startup)
                    holds.append(bound <= attempt.billed_output + 1e-6)
                    ratios.append(bound / max(attempt.billed_output, 1.0))
                rates = adjacent_rates(attempt)
                if rates is not None:
                    seen_rates.append(rates)
                seen_ttfb.append(attempt.ttfb)
        bounds[percentile] = {
            "n": len(holds),
            "holds": float(np.mean(holds)),
            "ratio_median": float(np.median(ratios)),
            "ratio_p90": float(np.percentile(ratios, 90)),
        }

    report = {
        "attempts": len(attempts),
        "duration_fit": {"startup_seconds": startup_d, "seconds_per_token": slope_d, "rate": 1 / slope_d, "r2": r2_d},
        "ttfb_fit": {"startup_seconds": startup_t, "seconds_per_token": slope_t, "rate": 1 / slope_t, "r2": r2_t},
        "session_rate": {
            "median": float(np.median(session_rates)),
            "q25": float(np.percentile(session_rates, 25)),
            "q75": float(np.percentile(session_rates, 75)),
            "min": float(min(session_rates)),
            "max": float(max(session_rates)),
        },
        "bound": bounds,
    }
    lines = [
        f"# Decode time on {runs_root.parent.parent.name}: {len(attempts)} attempts",
        "",
        "## The two relations the model implies",
        "",
        "| relation | startup | rate | R2 |",
        "|---|---|---|---|",
        f"| wall time = c + billed / r | {startup_d:.2f} s | {1 / slope_d:.0f} tok/s | {r2_d:.3f} |",
        f"| time to first byte = c + hidden / r | {startup_t:.2f} s | {1 / slope_t:.0f} tok/s | {r2_t:.3f} |",
        "",
        f"Decode rate read from adjacent chunks, per session: median {np.median(session_rates):.1f} tok/s, "
        f"quartiles {np.percentile(session_rates, 25):.1f} to {np.percentile(session_rates, 75):.1f}, "
        f"range {min(session_rates):.1f} to {max(session_rates):.1f}.",
        "",
        "## The inequality the model suggests",
        "",
        "Bound: billed >= rate x (wall time - startup), with startup the smallest time to first byte seen "
        "earlier in the session and rate a low percentile of the adjacent-chunk rates seen earlier.",
        "",
        "| rate percentile | n | share holding | bound / billed, median | 90th percentile |",
        "|---|---|---|---|---|",
    ]
    for percentile, row in bounds.items():
        lines.append(
            f"| P{percentile} | {row['n']} | {row['holds']:.3f} | {row['ratio_median']:.2f} | {row['ratio_p90']:.2f} |"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="How much of a call's wall time is decoding, and can silence bound the tokens?".splitlines()[0])
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "workspace" / "experiments" / "decode_bound")
    args = parser.parse_args()
    run(args.runs_root, args.output_dir)


if __name__ == "__main__":
    main()
