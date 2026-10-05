# Run a new task

Accept a project directory and task description. Current checkpoints cannot forecast
arbitrary projects. Offer collection without prediction; never invent a forecast or
convert the project into a benchmark pack.

For collection, call `prepare_task` with `project`, `task`, `collect_only=true`,
`binary`, `llm_model`, `agent_id` and a new `output` outside the source repository.
Use the Git root. The copy includes uncommitted edits, deletions and unignored new
files, with separate Git metadata and no remote. Linked files and submodules are
unsupported. Without collect_only, output prepares a copy without execution.

Poll `task_status` until `awaiting_confirmation`. Show **no prediction available**,
the task, copied workspace, resolved model, reasoning, sandbox and stopping limits.
Defaults are 900 seconds and 500,000 reported input+output tokens; interruption can
overshoot. Ask whether to execute and **wait for a new user reply**. After agreement,
call `start_task(task_id=..., confirmation_id=...)` using the returned confirmation ID.
Poll until terminal and report actual usage and label status. Collection has no
prediction ID or pairing. Do not call record_usage for it. Captures stay in output.
Keep changes in the copy for review; never apply them to the source automatically.

For a supported task pack:

1. Check `model_status`. Use the checkpoint's model, agent and execution condition.
2. Call `prepare_task` with `pack`, `binary`, `llm_model`, `agent_id` and a new
   `output` directory. It copies the workspace and creates a fresh Codex thread.
   It does not start the model turn. Repeating the same request retains the task.
3. Poll `task_status` while preparing, with a short pause between checks.
4. At `awaiting_confirmation`, show the task, point estimate, interval, token
   unit, prediction ID and saved time/token limits. Mention validation-only scope.
   Ask whether to execute and **wait for a new user reply**. An earlier request
   to estimate or prepare does not authorize execution.
5. After agreement, call `start_task` with the returned task and prediction IDs.
   Execution consumes model usage and edits the copied workspace. Poll status
   until terminal, then report actual usage, pairing and submission checks from
   the returned result. Flag a missing submission marker or submission error.

Use `cancel_task` if declined. Confirmation expires after 30 minutes. A server
restart invalidates pending tasks; prepare again and ask about the new forecast.
Never automatically retry failed, cancelled or expired runs. Keep their captures.
Cancellation during execution requests interruption; partial usage remains pending.

Preparation and execution use the same thread and prompt. Forecasts keep their
original checkpoint and prediction. A changed project copy requires preparation again.
Duplicate starts reuse the run. `failed` includes the underlying error in `result`.
Completion establishes the usage boundary; benchmark correctness needs separate tests.

For an explicitly requested terminal workflow, check prerequisites first:

```sh
tc runtime --model MODEL_DIR --data USAGE_DB run-codex --pack PACK_DIR --binary CODEX_BINARY --llm-model MODEL_ID --agent-id AGENT_ID --output NEW_CAPTURE_DIR --check
```

Removing `--check` executes immediately. Do not use this command to bypass the
chat confirmation flow. Limits come from the checkpoint and can overshoot during
interruption. The check makes no prediction and does not verify authentication.
