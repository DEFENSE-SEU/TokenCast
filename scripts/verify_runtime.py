"""Verify a real checkpoint and observation over stdio MCP, including restart pairing."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from token_prediction.api import read_object
from token_prediction.dataset import PredictionPoint, PredictionPosition, PredictionTarget
from token_prediction.estimators.base import RunContext
from token_prediction.estimators.lightgbm_bundle import load_lightgbm_bundle
from token_prediction.evaluation import load_fitted_calibrator


def native_forecast(model_path: Path, observation: dict) -> dict:
    """Measure repeated inference on the original point without changing its identifiers."""
    model = load_lightgbm_bundle(model_path)
    if observation.get("mode") == "codex":
        from token_prediction.collection.readers.codex_task import build_launch_event, launch_point

        point = launch_point(build_launch_event(observation["launch"], observation["prompt"]))
        context = RunContext(
            task_id=point.task_id,
            trajectory_id=point.trajectory_id,
            run_id=point.run_id,
            dataset_id=model.dataset_id,
            condition_id=point.condition_id,
            target=point.target,
            input_contract_id=model.input_contract_id,
            target_unit_contract_id=point.token_unit_contract_id,
            observation_unit_contract_id=point.token_unit_contract_id,
        )
    else:
        context_data = dict(observation["context"])
        context_data["target"] = PredictionTarget(context_data["target"])
        context = RunContext(**context_data)
        point_data = dict(observation["point"])
        point_data["position"] = PredictionPosition(point_data["position"])
        point_data["target"] = PredictionTarget(point_data["target"])
        point_data["features"] = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in point_data["features"].items()
        }
        point = PredictionPoint(**point_data)
    session = model.start(context)
    calibration_path = model_path.parent / "calibrator.json"
    calibrator = (
        load_fitted_calibrator(read_object(calibration_path)) if calibration_path.exists() else None
    )
    raw = session.predict(point)
    forecast = raw if calibrator is None else calibrator.transform(raw)
    samples = []
    for _ in range(20):
        started = time.perf_counter()
        result = session.predict(point)
        if calibrator is not None:
            result = calibrator.transform(result)
        samples.append((time.perf_counter() - started) * 1000)
        assert result == forecast, "native inference changed for the identical point"
    return {
        "forecast": {key: getattr(forecast, key) for key in ("lower", "point", "upper")},
        "raw_forecast": {key: getattr(raw, key) for key in ("lower", "point", "upper")},
        "warm_inference_ms": {
            "samples": samples,
            "count": len(samples),
            "median": statistics.median(samples),
            "p95": sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
            "p95_method": "nearest rank",
            "includes_fixed_calibrator": calibrator is not None,
        },
    }


async def verify(model: Path, observation: Path, actual: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    rounds = []
    first_prediction = None
    with tempfile.TemporaryDirectory(prefix="token-prediction-") as directory:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "token_prediction.mcp",
                "--model",
                str(model),
                "--data",
                str(Path(directory) / "usage.sqlite3"),
            ],
            env=env,
        )
        for round_index in range(2):
            started = time.perf_counter()
            async with stdio_client(parameters) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    report = {
                        "round": round_index + 1,
                        "initialize_ms": (time.perf_counter() - started) * 1000,
                        "calls": [],
                    }

                    async def call(name: str, arguments: dict, expected: str) -> dict:
                        before = time.perf_counter()
                        response = await session.call_tool(name, arguments)
                        elapsed = (time.perf_counter() - before) * 1000
                        value = response.structured_content
                        assert not response.is_error, value
                        assert value["status"] == expected, value
                        report["calls"].append(
                            {"tool": name, "roundtrip_ms": elapsed, "result": value}
                        )
                        return value

                    status = await call("model_status", {}, "ok")
                    report["cold_load_ms"] = status["load_ms"]
                    prediction = await call(
                        "predict_tokens",
                        {"observation_path": str(observation)},
                        "ok" if round_index == 0 else "duplicate",
                    )
                    if first_prediction is None:
                        first_prediction = prediction
                    else:
                        assert {**prediction, "status": "ok"} == first_prediction
                    actual_arguments = {
                        "prediction_id": prediction["prediction_id"],
                        "actual_path": str(actual),
                    }
                    await call(
                        "record_usage",
                        actual_arguments,
                        "matched" if round_index == 0 else "duplicate",
                    )
                    repeated = await call(
                        "predict_tokens", {"observation_path": str(observation)}, "duplicate"
                    )
                    assert {**repeated, "status": "ok"} == first_prediction
                    await call("record_usage", actual_arguments, "duplicate")
                    rounds.append(report)
    return {"rounds": rounds, "prediction": first_prediction}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "observation", "actual", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    observation = args.observation.resolve()
    actual = args.actual.resolve()
    model = args.model.resolve()

    async def bounded_verify() -> dict:
        async with asyncio.timeout(120):
            return await verify(model, observation, actual)

    result = asyncio.run(bounded_verify())
    native = native_forecast(model, read_object(observation))
    for name in ("forecast", "raw_forecast"):
        for key, value in native[name].items():
            assert math.isclose(result["prediction"][name][key], value, rel_tol=1e-12)
    result.update(
        {
            "verified": True,
            "model": str(model),
            "observation": str(observation),
            "actual": str(actual),
            "native": native,
            "scope_note": "MCP transport, persistence and native-model consistency check. "
            "Single-point timings provide no general accuracy or coverage evidence. "
            "Pre-turn timing and actual-task provenance require the original capture.",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"verified": True, "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
