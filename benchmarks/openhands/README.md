# OpenHands CodeAct collection

This directory contains a runnable collector for the legacy OpenHands
**CodeAct 0.62.0** harness. It accepts the same `task.md` and seed workspace
from a task pack used by DSH, then launches one fresh OpenHands container. The
collector copies only the seed into the actor workspace; runner oracles remain
outside the container. It records native trajectory and provider wire evidence
under one run directory. Raw runs are written below ignored `workspace/` paths.

Supported task benchmarks include SWE-bench, Search-R1, MMLU-Pro, and
LongBench-v2. Check the model configuration and wire records when comparing
captures from different collections.

## Requirements

- Docker with a running Linux engine and the Git-equipped image built from
  `Dockerfile.git`. The collector checks OpenHands version `0.62.0` and Git
  before any provider request.
- An OpenAI-compatible chat-completions endpoint. Set
  `OPENHANDS_UPSTREAM_API_KEY` in the host environment for authenticated
  endpoints. For a local endpoint that accepts no credential, use
  `--allow-unauthenticated-upstream`.
- A task pack with `pack.json`, a UTF-8 task statement, and a present seed
  workspace. Downloaded packs under `benchmarks/dsh/` are ignored by Git and
  must be prepared separately. A pack with an obsolete absolute seed path must
  be rebuilt before collection.

Build the pinned image, then inspect a task without making a provider request:

```bash
docker build -f benchmarks/openhands/Dockerfile.git \
  -t token-prediction-openhands:0.62-git .
python -m benchmarks.openhands.collect check \
  --pack benchmarks/dsh/repo_repair/low_catalog_key \
  --model openai/YOUR_MODEL_ID \
  --upstream-base-url https://YOUR_ENDPOINT/v1
```

Run one task after setting the endpoint's key in the host environment:

```bash
python -m benchmarks.openhands.collect run \
  --pack benchmarks/dsh/repo_repair/low_catalog_key \
  --model openai/YOUR_MODEL_ID \
  --upstream-base-url https://YOUR_ENDPOINT/v1 \
  --run-dir workspace/runs/openhands/low-catalog-key-r0 \
  --authorize-provider-requests
```

The collector refuses an existing run directory. `--max-iterations`,
`--timeout-seconds`, `--max-input-tokens`, `--max-output-tokens`,
`--num-retries`, and `--checkpoint-bytes` set explicit limits. The defaults are
500 iterations, 7,200 seconds, 262,144 input tokens, 65,536 output tokens,
four retries, and 128 bytes per checkpoint. These limits are recorded in the
run metadata; the collector currently enforces iteration and wall-clock caps.

The official app image omits Git in its final stage. The derived image adds it
for workspace initialization and coding tasks. `--image` selects another image
that passes the same version and Git preflight.

Each run saves `run.json`, the exact `task.txt` bytes and SHA-256, `workspace/`,
`config.toml`, `console.log`, `native/trajectory.json` when OpenHands emits it,
`observer.jsonl`, `wire/<attempt-id>/`, and `terminal.json`. The host proxy
keeps the real provider key in memory; the container receives a temporary
proxy-only key. Wire records save request JSON, response bytes, stream timing,
provider usage when returned, and byte checkpoints over text, reasoning, and
tool arguments. Missing usage remains marked missing.

The proxy observes HTTP requests and attempts. It cannot identify which
requests are OpenHands retries of one logical call, so `logical_calls` is null
and `retry_grouping` is `unavailable` in the summary. It does not emit inferred
task-update events. Analyses requiring the exact four-position call hierarchy
must establish that grouping from additional runtime evidence.

OpenHands 0.62.0's [headless CLI](https://github.com/OpenHands/OpenHands/blob/0.62.0/openhands/core/config/arg_utils.py)
accepts the task file, CodeAct agent, and iteration cap. Its
[CLI runtime](https://github.com/OpenHands/OpenHands/blob/0.62.0/openhands/runtime/impl/cli/cli_runtime.py)
runs in the mounted task workspace, and the
[app image](https://github.com/OpenHands/OpenHands/blob/0.62.0/containers/app/Dockerfile)
contains the pinned runtime.
