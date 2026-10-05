"""Predict and pair a saved real observation; this does not run an LLM task."""

import json
from pathlib import Path

from token_prediction.api import TokenCast, read_object

example = Path(__file__).resolve().parent
service = TokenCast(example / "model/bundle", example.parents[1] / "workspace/example-usage.sqlite3")
prediction = service.predict_tokens(read_object(example / "observation.json"))
if prediction["status"] not in {"ok", "duplicate"}:
    raise SystemExit(json.dumps(prediction))
pairing = service.record_usage(prediction["prediction_id"], read_object(example / "actual.json"))
if pairing["status"] not in {"matched", "duplicate"}:
    raise SystemExit(json.dumps(pairing))
print(json.dumps({"mode": "saved_observation", "forecast": prediction["forecast"],
                  "inference_ms": prediction["inference_ms"], "pairing": pairing}, indent=2))
