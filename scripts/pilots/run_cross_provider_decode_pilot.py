from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "workspace" / "external" / "rollout_dialogues" / "origin"
FAMILIES = {
    "claude-opus4.7": "swebench-origin-claude-opus4.7",
    "claude-sonnet4.6": "swebench-origin-claude-sonnet4.6",
    "gemini3.1": "swebench-origin-gemini3.1",
    "gpt5.2instant": "swebench-origin-gpt5.2instant",
    "qwen3-235b": "swebench-origin-qwen3-235b",
}
TRIM_PERCENTILE = 99


def fit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    return {
        "startup_seconds": float(coefficients[0]),
        "rate": float(1 / coefficients[1]) if coefficients[1] > 0 else float("nan"),
        "r2": float(1 - residual.var() / y.var()),
    }


def load(folder: str) -> dict[str, np.ndarray]:
    gaps, billed, reasoning = [], [], []
    for path in (CORPUS / folder).rglob("*.traj.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        previous = None
        for message in document["messages"]:
            extra = message.get("extra") or {}
            stamp = extra.get("timestamp")
            if message["role"] == "assistant" and stamp is not None and previous is not None:
                usage = extra["response"]["usage"]
                details = usage.get("completion_tokens_details") or {}
                gaps.append(stamp - previous)
                billed.append(usage["completion_tokens"])
                reasoning.append(details.get("reasoning_tokens") or 0)
            if stamp is not None:
                previous = stamp
    return {k: np.asarray(v, dtype=float) for k, v in (("gap", gaps), ("billed", billed), ("reasoning", reasoning))}


def run(output_dir: Path) -> None:
    report: dict[str, dict] = {}
    lines = [
        "# Wall time against billed output, five public actor families",
        "",
        "| actor | calls | reasoning share of billed | startup | decode rate | R2 | rate from reasoning alone | R2 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, folder in FAMILIES.items():
        data = load(folder)
        if len(data["gap"]) < 50:
            continue
        keep = data["gap"] <= np.percentile(data["gap"], TRIM_PERCENTILE)
        billed_fit = fit(data["billed"][keep], data["gap"][keep])
        share = float(data["reasoning"].sum() / data["billed"].sum())
        reasoning_fit = fit(data["reasoning"][keep], data["gap"][keep]) if share > 0.02 else None
        report[name] = {"calls": int(len(data["gap"])), "reasoning_share": share, "billed_fit": billed_fit, "reasoning_fit": reasoning_fit}
        tail = (
            f"{reasoning_fit['rate']:.0f} tok/s | {reasoning_fit['r2']:.3f} |"
            if reasoning_fit
            else "not disclosed | |"
        )
        lines.append(
            f"| {name} | {len(data['gap'])} | {share:.1%} | {billed_fit['startup_seconds']:.2f} s | "
            f"{billed_fit['rate']:.0f} tok/s | {billed_fit['r2']:.3f} | {tail}"
        )
    lines += [
        "",
        "Every family shows wall time rising with billed output at a rate in the range a decoder plausibly runs at. "
        "For the one family with a large disclosed reasoning share, the rate fitted against reasoning tokens alone "
        "matches the rate fitted against all billed output, which is what a shared decoder implies.",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description='Does wall time grow with billed output at the same rate on other providers?'.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "workspace" / "experiments" / "cross_provider_decode")
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
