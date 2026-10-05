from __future__ import annotations

import json
import math
from collections import Counter

from scripts.experiments.fit_exgentic_v2_context_exposure import (
    FEATURE_SOURCE, LOCK, MANIFEST, METHODS, OUTPUT, RECEIPT, ROOT, ROSTER,
)
from scripts.experiments.predict_exgentic_v2_context_exposure import (
    CLOCK_FILE, FIT_SCRIPT, PREDICTIONS, PREDICTION_MANIFEST,
)
from scripts.experiments.run_public_trajectory_validation import _sha256


PREDICT_SCRIPT = ROOT / "scripts/experiments/predict_exgentic_v2_context_exposure.py"


def main() -> int:
    manifest = json.loads(PREDICTION_MANIFEST.read_text(encoding="utf-8"))
    receipt_file = OUTPUT / "fit_receipt.json"
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    checks = {
        "source_lock_sha256": _sha256(LOCK),
        "download_receipt_sha256": _sha256(RECEIPT),
        "roster_sha256": _sha256(ROSTER),
        "split_manifest_sha256": _sha256(MANIFEST),
        "feature_source_sha256": _sha256(FEATURE_SOURCE),
        "fit_script_sha256": _sha256(FIT_SCRIPT),
        "fit_receipt_sha256": _sha256(receipt_file),
        "prediction_script_sha256": _sha256(PREDICT_SCRIPT),
        "clock_lookup_sha256": _sha256(CLOCK_FILE),
        "prediction_sha256": _sha256(PREDICTIONS),
    }
    if any(manifest[key] != value for key, value in checks.items()):
        raise ValueError("frozen source/code/prediction hash mismatch")
    if manifest["model_sha256"] != receipt["model_sha256"]:
        raise ValueError("prediction manifest model hashes differ from fit receipt")
    for head, digest in receipt["model_sha256"].items():
        if _sha256(OUTPUT / "models" / f"{head}.txt") != digest:
            raise ValueError(f"model file changed: {head}")
    roster = {row["session_id"]: row for row in
              (json.loads(line) for line in ROSTER.read_text(encoding="utf-8").splitlines())}
    test = {session: row for session, row in roster.items() if row["split"] == "test"}
    if len(test) != 1338:
        raise ValueError("test roster size changed")
    observed = Counter()
    terminal = Counter()
    by_benchmark = Counter()
    last = None
    for line in PREDICTIONS.open(encoding="utf-8"):
        row = json.loads(line)
        session, k = row["session_id"], row["k"]
        if session not in test:
            raise ValueError("non-test session in prediction file")
        source = test[session]
        if (row["family_key"] != source["family_key"] or row["benchmark"] != source["benchmark"]
                or row["model"] != source["model"] or row["harness"] != source["harness"]):
            raise ValueError("prediction identity metadata differs from roster")
        if not 1 <= k <= source["observed_successful_calls"]:
            raise ValueError("invalid prefix index")
        if last is not None and (session, k) <= last:
            raise ValueError("prediction rows are duplicated or unsorted")
        last = (session, k)
        if row["terminal"] != (k == source["observed_successful_calls"]):
            raise ValueError("terminal marker incorrect")
        if set(row["predictions"]) != set(METHODS):
            raise ValueError("method set changed")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
               for value in row["predictions"].values()):
            raise ValueError("prediction is invalid")
        observed[session] += 1
        terminal[row["terminal"]] += 1
        by_benchmark[row["benchmark"]] += 1
    if any(observed[session] != row["observed_successful_calls"] for session, row in test.items()):
        raise ValueError("prediction prefixes incomplete")
    if terminal != {False: 31764, True: 1338}:
        raise ValueError("terminal/nonterminal count differs")
    if (len(observed) != manifest["test_sessions"] or sum(observed.values()) != manifest["prediction_rows"]
            or dict(by_benchmark) != manifest["rows_by_benchmark"]):
        raise ValueError("prediction manifest counts differ")
    if manifest["test_suffix_targets_computed"] or manifest["test_scores_computed"]:
        raise ValueError("prediction manifest claims test scoring")
    print(json.dumps({"read_only_verified": True,
                      "test_sessions": len(observed), "prediction_rows": sum(observed.values()),
                      "nonterminal_rows": terminal[False], "terminal_rows": terminal[True],
                      "prediction_sha256": checks["prediction_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
