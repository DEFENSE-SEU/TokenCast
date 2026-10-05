# TokenCast plugin

Local token prediction through a Skill and MCP tools. Supports LightGBM
task-launch checkpoints for recorded DSH trajectories and controlled Codex tasks.

## Setup

Run from the repository root in your Python environment:

```sh
python -m pip install ".[runtime,mcp]"
codex plugin marketplace add .
codex plugin add token-prediction@tokencast-local
```

Set TOKEN_PREDICTION_MODEL to your bundle directory and TOKEN_PREDICTION_DATA to
a writable SQLite file, using absolute paths for both before starting Codex.
Keep the supplied calibrator.json beside the bundle directory. The client must
use the Python environment where the package is installed.

The repository includes a validation checkpoint at examples/codex/model/bundle.
Run `python examples/codex/run.py` from the repository root to check it locally.

In a new chat, invoke $predict-tokens with a supported task pack or observation.
For a new task, review the preview and reply to confirm execution. Project +
description supports collection without a forecast: use collect_only to prepare
an independent copy, including uncommitted changes and unignored new files.
Execution waits for your reply. Defaults: 15 minutes and 500,000 input+output tokens.
Collected usage stays with the capture; no prediction or pairing is created.

## Tools

| Tool | Purpose |
| --- | --- |
| model_status | Check the model and supported inputs. |
| predict_tokens | Predict from an observation object or JSON file. |
| record_usage | Pair measured usage with a prediction ID. |
| prepare_task | Prepare a forecast or collection-only preview without executing. |
| task_status | Read the forecast, progress and final usage. |
| start_task | Execute the prepared task after your confirmation. |
| cancel_task | Cancel preparation or request interruption. |

For prediction and usage recording, supply either the object or its file path.
Usage must match the saved prediction's scope and token unit. Repeating an
identical request returns the saved record.

## Command line

```sh
tc runtime --model MODEL_DIR --data usage.sqlite3 status
tc runtime --model MODEL_DIR --data usage.sqlite3 predict --input observation.json
tc runtime --model MODEL_DIR --data usage.sqlite3 record-usage --prediction-id PREDICTION_ID --input actual.json
```

To start the MCP server directly:

```sh
python -m token_prediction.mcp --model MODEL_DIR --data usage.sqlite3
```

The model is reused within a running MCP process. Pending confirmations expire
after 30 minutes or a server restart. CLI run-codex executes immediately.

## Scope

The loaded checkpoint defines supported tasks and model conditions. Native
Codex prediction covers a fresh thread, one turn and the main agent's reported
input plus output tokens. Current checkpoints do not support Claude tasks or
automatic prediction of an ongoing IDE conversation. No Hooks, online model
updates or cost conversion are included.
