---
name: predict-tokens
description: Use TokenCast to check a local model, predict token consumption from supported observations, run a supported Codex task, or pair measured usage.
---

# TokenCast

Call `model_status` first. Check readiness, supported modes, conditions and
`deployment_scope`. Report missing inputs or unsupported scope without inventing
features or changing task metadata. Validation-only checkpoints provide no
general accuracy or interval coverage guarantee.

- **New task:** follow [Run a new task](references/new-task.md). Accept a project
  directory and description, or a supported pack. Show the prepared forecast or
  explicit collection-only preview, including its execution limits, and
  wait for the user's reply before calling `start_task`. An ongoing chat cannot
  supply a fresh-task observation.
- **Saved observation:** call `predict_tokens` with exactly one of `observation`
  or `observation_path`. Preserve its data and provenance. DSH uses `mode=replay`;
  native Codex uses `mode=codex`. Replay does not establish a new pre-turn forecast.
- **Actual usage:** call `record_usage` with the prediction ID and exactly one of
  `actual` or `actual_path`. Match the saved scope and token unit.

For forecasts, report the point estimate, interval, inference time and prediction ID.
Collection-only tasks return actual usage without a forecast or pairing.
`duplicate` retains the original result and timing; `pending` saves no observed
actual. Explain other statuses using the returned error. Keep the model and
calibrator fixed; do not estimate missing counts or convert tokens to cost.

Native usage is one fresh thread, one turn and the main agent's reported input
plus output. Cached input and reasoning output are subsets. Resumes, forks,
subagents and unreported retries are outside scope. Failed zero counters are
not observed usage; missing, censored or invalid labels require a null value.

Paths resolve on the MCP server. Forecasts need the runtime, a model and a SQLite
path. Collection needs Codex and a new output directory, without a prediction model.
It does not collect the current chat. Native Codex checkpoints do not
accept Claude observations.
