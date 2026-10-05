# Codex example

From the repository root, after installing `.[runtime,mcp]`:

```sh
python examples/codex/run.py
```

This replays the saved input from a real Flask task and pairs its recorded usage.
It needs no Codex login or LLM call. Expected point: about 243,393 tokens; actual:
263,292 input+output tokens. Repeating it returns the same stored record.

The unchanged LightGBM checkpoint is validation-only: 8 fitting tasks, 3 validation
tasks and 8 calibration tasks. It covers SWE-bench source repairs with Codex CLI
0.157.0, gpt-6-astra, low reasoning and workspace-write, one fresh main-agent turn.
It has no established general-project accuracy or interval coverage guarantee.
The model and calibrator retain checkpoint ID
`951134a14f5bd45d997b29ac8490f1cecbb630bb99cdc25afea8fec8cceecbd6`.
This repeated Flask run is a workflow example, not an additional independent test.

## Prepare a new run

This new-task example requires PowerShell and has been validated on Windows.

Fetch the pinned source checkout (Git and network required):

```sh
git init workspace/example-flask
git -C workspace/example-flask remote add origin https://github.com/pallets/flask.git
git -C workspace/example-flask fetch --depth 1 origin 7ee9ceb71e868944a46e1ff00b506772a53a4f1d
git -C workspace/example-flask checkout --detach FETCH_HEAD
```

The pack expects workspace/example-flask. If it already exists, verify its pinned
commit and clean status before reuse. Replace CODEX_BINARY with your native Codex
executable path and check the local runtime without executing:

```sh
tc runtime --model examples/codex/model/bundle --data workspace/example-usage.sqlite3 run-codex --pack examples/codex/pack --binary "CODEX_BINARY" --llm-model gpt-6-astra --agent-id codex --output workspace/example-new-run --check
```

Set TOKEN_PREDICTION_MODEL to the absolute path of examples/codex/model/bundle.
Give the Skill absolute paths for examples/codex/pack, your Codex executable and
a new output directory, with model gpt-6-astra and agent codex. Review the forecast
before confirming execution. A different client/model condition is rejected.
This needs an authenticated Codex client and consumes model usage after confirmation.
Task correctness requires separate benchmark tests; this package includes no oracle.

## Source and terms

The task is SWE-bench Verified instance pallets__flask-5014, derived from
[pallets/flask #5014](https://github.com/pallets/flask/pull/5014).
The snapshot input and usage were captured by TokenCast on October 5, 2026.
The model and TokenCast-authored example files use the repository AGPL-3.0
license. The included SWE-bench notice preserves its upstream MIT terms;
the fetched Flask checkout retains its own license.
