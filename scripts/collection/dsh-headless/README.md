# DSH headless profile

This package adds preset-aware one-shot and persistent batch runners to a
dedicated DSH profile.
The runner mounts the requested Agent preset before Agent publication, freezes
provider, model, reasoning, temperature, output cap, and stop strings on each
Request, and leaves session JSONL plus the Token observer sidecar for the DSH
ledger.

The tracked `router-standard` preset is the only system-trusted preset root.
User preset and bundled-skill roots are excluded. The profile disables title
generation, settings reload, compaction, web, delegation, telemetry, and every
auxiliary model path; it also fixes the native 12-tool actor surface. The actor
uses `unrestricted` with approval policy `never` so the Windows task packs
can run nested test processes. Before formal collection, pre-create the profile
module fallback links and use a worker identity that can modify actor
workspaces, output roots, and the generated profile `cordis.yml`. Keep the
profile manifest, patch, module-link targets, integration, and preset content
read-only to that worker. This operating-system boundary must be deployed and
verified before collection authorization.

Set `DSH_BATCH=1` to keep one DSH process alive across sequential jobs.
Standard output becomes a newline-delimited JSON channel with `ready`,
`job/started`, `job/done`, `job/error`, `fatal`, and `bye` frames. The `ready`
frame declares batch protocol version 2. Every `job` supplies its own session
ID, cwd, task, decoding condition, and collection caps. The runner creates a
fresh Agent handle for that job, reports the exact physical session path,
flushes the session, disposes the handle, and then accepts the next job. A
failed job follows the same disposal boundary and does not close the batch
service. A persistence or disposal failure emits `fatal` and exits so the
caller can start a fresh process. Protocol v2 error and fatal frames include
`request_boundary_crossed`, which records whether the job crossed its
`agent/request` boundary.

The batch service reads the same newline-delimited JSON framing from standard
input. DSH's headless startup still expects one positional task; use an inert
value such as `batch`, then send jobs on standard input. Finish by sending
`{"type":"shutdown"}` and wait for `bye` and process exit.

An optional `budget_shadow` object on a batch job records a dry-run stop
decision at each `llm/stream` Attempt. It requires `output_path` (a new absolute
JSONL file), `source_id`, `task_family`, `run_id`, `budget_tokens`, and
`margin_tokens`. The source and family fields are required for each job so
sequential batch tasks do not share an unidentified trace. The sidecar records
DSH's request-time token-meter estimate, already settled provider-accounted
tokens, the threshold decision, retry ordinal, and the provider's terminal
input/output/cache usage. A flagged decision still dispatches the Request.
The shadow threshold uses `settled + max(0, DSH meter estimate - margin)`;
it is an instrumentation baseline for DSH and does not reuse the EnvBench
transport predictor. A join with the existing native session and observer
records is needed to reconstruct prior-call and tool-result features.
The token-meter value is an estimate and the `yuan_spend` field stays null
without a verified tariff. The final provider body may differ from DSH's
request surface, so this trace needs a route-specific calibration before it
can justify a live stop policy. Ordinary jobs omit `budget_shadow` and keep the
existing behavior. `node --test test/budget-shadow.test.mjs` checks the local
middleware without any provider call. The Python collector does not yet send
this optional field, and no live DSH provider run has verified the hook.

Model-generated session titles are disabled in this profile, so every provider
call belongs to an observed Agent Call. The frozen manifest supplies the hosted
retry limit through `DSH_MAX_RETRIES`; the current SWE-bench protocol permits
four retries, or five observable Attempts, per logical Call. The `(turn, step)`
pair is the logical Call key. Repeated Requests with the same key remain retries
within that Call, consume one logical-Call cap entry, and retain separate
observer Attempts. Before starting the next Call, the runner checks the task's
logical-Call limit, cumulative provider-accounted Token limit, and complete
Attempt usage.

The profile uses DSH `0.1.1-rc.2` with Zstandard-compressed session logs,
chunk packing enabled, a 200 ms persistence batch window, and an explicit
ten-call parallel tool ceiling. Every invocation owns one
fresh session and `AgentHandle`; in batch mode, this ownership boundary applies
to each job. It flushes the terminal session prefix, then awaits handle
disposal. The free preflights additionally export a decoded
`session.jsonl` view for contract checks; collection retains the physical
`session.jsonl.zstd` artifact. A session has one live persistence writer.

The deterministic adapter is enabled only when `DSH_FIXTURE=1`. It makes no
network request. Its first Attempt ends with `DSH_FIXTURE_RETRY`; DSH's own
provider retry policy creates one second Attempt in the same logical Call, and
the second Attempt returns terminal usage and `fixture complete`.

The local actor is enabled only when `DSH_LOCAL_QWEN=1`. It runs the frozen
Qwen2.5-7B-Instruct snapshot offline with NF4 weights on CUDA, streams visible
text into DSH before generation completes, and uses one ACK per published
delta so the observer commits online checkpoints before the next model chunk.
In batch mode, one local worker and its loaded model serve sequential DSH jobs.
Each job still owns a fresh DSH session and Agent handle. Generation Probe
staging files are scoped by session ID, and the worker closes the active file
before publishing each Request terminal so Windows collectors can move it
immediately. The collector retains the stable final artifact name
`generation-probe.jsonl`.

Copy [`.env.example`](../../../.env.example) to `.env`, select a connection
profile, and pass the file to the Python collector with `--env-file`.
The file uses `NAME=VALUE` with full-line comments, without shell evaluation,
variable expansion, or duplicate names. Filled environment files are ignored.

Public connection settings are:

| Variable | Meaning |
| --- | --- |
| `MODEL_PROVIDER` | Connection profile: `openai`, `qwen`, `gemini`, `kimi`, `claude`, `deepseek`, or `openai-codex`. |
| `<PREFIX>_API_KEY` | Key for the selected profile, e.g. `OPENAI_API_KEY` or `KIMI_API_KEY`. |
| `<PREFIX>_BASE_URL` | Endpoint for that profile, e.g. `OPENAI_BASE_URL`. Claude requires a Chat Completions gateway; native Anthropic API is unsupported. |
| `MODEL_CONTEXT_WINDOW` | The endpoint's context capacity. |
| `MODEL_MAX_OUTPUT_TOKENS` | Output ceiling, at least the manifest's `max_attempt_output_tokens`. |
| `OPENAI_CODEX_AUTH_FILE` | Optional absolute Codex `auth.json` path for `openai-codex`. |

Provider precedence is `--provider` > process environment > `--env-file`.
Other settings prefer process values; public names beat legacy aliases within
the same source. Both public profile names and historical route names work
with `--provider`; existing `DSH_*` environment files remain supported.
When changing a public profile, use its named endpoint and key fields;
lower-priority unscoped legacy connection values are not inherited.

The profile selects connection settings. A shared `openai` gateway can serve
Gemini, Kimi, and other model families. Model IDs and decoding stay fixed by
the manifest, so `*_MODEL` values in `.env` do not select collection models.
The collector translates public settings into the internal DSH adapter fields.
Direct Node invocations still use those internal fields.

The `qwen-compatible` route is DSH's generic pi-ai adapter speaking the
OpenAI-compatible Chat Completions protocol. The route id is frozen into the
collection manifests and their condition digests, so it keeps its historical
name. Its request mapping follows the selected model family: Qwen's `off`
condition sends `enable_thinking=false`, Kimi K2.5's `off` condition sends
`thinking.type=disabled`, and Gemini forwards a supported `reasoning_effort`.
Both the request timeout and stream-idle timeout are five minutes so normal
hosted startup latency does not become a failed Attempt.

Run an offline configuration check after filling the template. It needs no
DSH installation, prints no secret values, and creates no collection:

```powershell
python -m scripts.collection.run_dsh check-config `
  --manifest .\configs\collection\development_collection_v16.json `
  --env-file .\.env
```

The tracked
[`development_sufy_gemini_kimi_v1.json`](../../../configs/collection/development_sufy_gemini_kimi_v1.json)
contains two gateway conditions:

| Model ID | Frozen reasoning setting | Wire control |
| --- | --- | --- |
| `google/gemini-3.5-flash` | `low` | `reasoning_effort: "low"` |
| `moonshotai/kimi-k2.5` | `off` | `thinking: {"type": "disabled"}` |

Kimi K2.5 also accepts `high` as the collection label for binary thinking
enabled (`thinking.type=enabled`); it has no graded effort control here.
Its reasoning content is retained in assistant tool history. Gemini tool-call
thought signatures, when returned, are preserved across streaming and replay.
Gemini 3 conditions must specify an enabled effort; `off` is rejected before
a provider request. Existing Qwen/GPT condition mappings remain unchanged.

Google documents `low` as a reduced thinking setting in its
[OpenAI compatibility guide](https://ai.google.dev/gemini-api/docs/openai#thinking).
Moonshot documents the `disabled` switch as
[Kimi K2.5 Instant mode](https://github.com/MoonshotAI/Kimi-K2.5#6-model-usage).
The manifest keeps temperature 0 from the existing development protocol;
Kimi's recommended Instant-mode temperature is 0.6. These vendor documents
describe parameter semantics; the configured gateway must support the model
IDs and controls itself.

Set the endpoint and key through `.env`, then inspect both conditions:

```powershell
python -m scripts.collection.run_dsh check-config `
  --manifest .\configs\collection\development_sufy_gemini_kimi_v1.json `
  --provider openai `
  --env-file .\.env
```

The 12-run schedule is for development diagnostics, excluded from paper
results. The inherited 64,000-token attempt cap and full benchmark collection
remain unverified. Offline configuration checks and short 512-token provider
smokes have narrower validation scope.

After installing the profile and preparing the manifest's task packs,
initialize the ledger before starting provider requests:

```powershell
python -m scripts.collection.run_dsh init `
  --manifest .\configs\collection\YOUR_FROZEN_MANIFEST.json `
  --run-root .\workspace\collections\hosted-swebench

python -m scripts.collection.run_dsh run-hosted-all `
  --manifest .\configs\collection\YOUR_FROZEN_MANIFEST.json `
  --run-root .\workspace\collections\hosted-swebench `
  --env-file .\.env `
  --authorize-commercial
```

Generated captures and dataset artifacts stay under `workspace/`, which is
ignored. Collection code, schemas, manifests, and tests remain tracked so the
dataset construction can be reviewed and reproduced.

The subscription actor is selected with `MODEL_PROVIDER=openai-codex`.
It exposes exactly the selected pi-ai catalog
model; the default is `gpt-5.4`. Responses SSE parsing, tool calls, terminal
usage, and retry Attempt boundaries stay in the audited pi-ai and DSH layers.
An empty stop list is omitted because the pi-ai adapter rejects the `stop`
field whenever it is present.

The adapter reads the existing Codex `auth.json` directly for every Request.
The collector accepts `OPENAI_CODEX_AUTH_FILE` and maps it to the adapter's
internal auth path; defaults are `$CODEX_HOME/auth.json`, then the current
user's `.codex/auth.json`. It strictly validates ChatGPT auth
mode, all required token fields, JWT expiry, and account consistency. Tokens
never enter environment variables, DSH credential storage, logs, receipts, or
capture artifacts. The file stays read-only. Codex remains the sole OAuth
refresh owner. An expired access token fails before pi-ai can call the refresh
endpoint; refresh the Codex sign-in and retry the Request.

Run the offline readiness check first. It reads authentication metadata and
the installed model catalog, makes no network request, and prints no account
identifier or token:

```powershell
node .\scripts\collection\dsh-headless\run-subscription-preflight.mjs
```

The result keeps `provider_access` at `unverified`. A successful check does not
establish backend entitlement until an explicitly authorized request succeeds.

An alternate auth file can be supplied as the first absolute-path argument.
The following low-level example invokes Node directly with internal `DSH_*`
fields. Normal collection uses the public `.env` settings above. After
readiness passes, run this authorized request from the repository root:

```powershell
$runRoot = Join-Path ([IO.Path]::GetTempPath()) ('token-prediction-subscription-' + [guid]::NewGuid().ToString('N'))
$env:DSH_OBSERVER_DIR = Join-Path $runRoot 'observer'
$env:DSH_SESSION_DIR = Join-Path $runRoot 'sessions'
$env:DSH_PRESET_ROOT = (Resolve-Path .\scripts\collection\dsh-headless\preset-root).Path
New-Item -ItemType Directory -Path $env:DSH_OBSERVER_DIR,$env:DSH_SESSION_DIR | Out-Null
$env:DSH_BATCH = '0'
$env:DSH_LOCAL_QWEN = '0'
$env:DSH_FIXTURE = '0'
$env:DSH_AGGREGATOR = '0'
$env:DSH_PROVIDER = 'openai-codex'
$env:DSH_MODEL = 'gpt-5.4'
$env:DSH_EXCLUSIVE_PROVIDER = 'openai-codex'
node "$env:APPDATA\npm\node_modules\@deepseek-ai\dsh\lib\bin.js" --profile dsh-headless 'your task'
```

The observer and compressed session artifacts remain under `$runRoot` for
inspection after the process exits.

This subscription route has explicit development manifests, including
`configs/collection/development_gpt_oauth_v19.json`. It is absent from the
original `dsh_p0_design_v1.json`. pi-ai `0.82.1` receives the Request
`maxTokens` value from DSH, while its Codex body builder omits a wire
`max_output_tokens` field. The route therefore lacks a hard wire-level
per-Attempt output ceiling. Keep this capability limit in the interpretation
of those runs. Provider-backed collection still requires the explicit
`--authorize-commercial` flag.

The Python collector uses this persistent transport for `run-all`. One DSH
process and one local model worker serve the sequential jobs, while every job
keeps its own Agent handle, session ID, observer sidecar, session log, and
Generation Probe capture. Batch staging lives under `<run-root>/batch`. A
`job/error` frame is recorded for that row and collection continues with the
same process. A `fatal` frame retires the process; the following job starts a
fresh process. At normal completion, the collector sends `shutdown`, waits for
`bye`, and verifies a zero process exit.

Local and hosted collection commands share the same persistent staging and
recovery path. Only one mutating collector owns a run root. On restart, the
collector refuses to proceed while a previous process tree remains alive. The
next invocation reconciles artifacts by the frozen session ID. Local rows use a
complete observer/session/Probe set; hosted rows use observer/session. Either
set finishes without another model call. Partial provider evidence is retained
as incident material and the affected row is excluded. Unrelated session files
remain in staging. Hosted batch collection retains frozen schedule order and
reuses a process across adjacent rows that share one model route. Every
observer Attempt route is checked against its frozen provider, model, and
reasoning condition before ledger projection.

A torn final JSONL record in any recovered row artifact quarantines the
recovered set in that row's capture as ambiguous provider evidence. The row is
excluded and replay is suppressed. Recovery preserves the durable prefix and
does not synthesize an observer Attempt terminal.

Before publishing a row terminal, the runner requires a participating session
flush. The observer's listener makes this flush an `fsync` barrier for dirty
sidecar records. The runner then disposes the row's Agent handle and waits for
that session's persistence retirement. An observer write or barrier failure
poisons completion and produces a batch `fatal`; a completion frame therefore
identifies durable session and observer artifacts.

For Generation Probe capture, the worker retains the current last-layer
post-state and computes a fixed 64-dimensional, seed-729 Rademacher projection
when a checkpoint is committed. A process-local weak-key bridge returns checkpoint identity
after the observer writes the Attempt checkpoint. The worker synchronously
appends and flushes `generation-probe.jsonl` before sampling the next Token.
Each checkpoint row records `source_output_tokens`, the original model-output
Token ordinal represented by that post-state. Observer-visible UTF-8 bytes keep
their separate role as the checkpoint cadence and four-field join coordinate.
Every routed Request is restricted to the frozen 12-tool DSH allowlist, which
excludes delegation, goal continuation, interactive, router-control, and web
tools. Guidance is retained only for tools present in each Request, followed by
one frozen DSH tool-surface instruction. The same names form an agent-scoped DSH
runtime restriction, keeping prompt, schemas, and global-tool lookup aligned.
The repository exposes a `qwen` Python extra for the compatible GPU dependency
ranges. Local preflight receipts record the model revision, NF4
configuration, and runtime tensor facts; they do not bind Python package
versions.

The collection runtime is pinned to DSH `0.1.1-rc.2`. Plugin installation also
requires pnpm. Initialize the default profile without a model call, copy it
only when `dsh-headless` is absent, then add the observer followed by this package:

```powershell
npm install --global pnpm @deepseek-ai/dsh@0.1.1-rc.2
$dshHomePath = if ($env:DSH_HOME) { $env:DSH_HOME } else { "$env:USERPROFILE\.dsh" }
$sourceProfile = Join-Path $dshHomePath 'profiles\headless'
$collectionProfile = Join-Path $dshHomePath 'profiles\dsh-headless'
if (-not (Test-Path -LiteralPath $collectionProfile)) {
  if (-not (Test-Path -LiteralPath (Join-Path $sourceProfile 'package.json'))) {
    dsh --profile headless --dump-default-config | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'DSH profile initialization failed' }
  }
  Copy-Item -LiteralPath $sourceProfile -Destination $collectionProfile -Recurse
}
dsh plugin --profile dsh-headless add "C:\path\to\TokenCast\scripts\collection\dsh-token-observer"
dsh plugin --profile dsh-headless add "C:\path\to\TokenCast\scripts\collection\dsh-headless"
```

Run the free preflight into a new output directory:

```powershell
node .\scripts\collection\dsh-headless\run-preflight.mjs C:\short\dsh-preflight
```

Run the installed-profile fixture batch transport smoke into another new
directory:

```powershell
node .\scripts\collection\dsh-headless\run-batch-preflight.mjs C:\short\dsh-batch-preflight
node .\scripts\collection\dsh-headless\run-batch-preflight.mjs --verify-existing C:\short\dsh-batch-preflight
```

The fixture batch transport smoke removes every casing of the hosted credential
key from its child environment and selects the deterministic fixture as the
exclusive provider. It requires `ready` protocol 2 before sending job 1. After
each `job/done`, it reads the complete observer and session JSONL, decodes packed
storage rows, and checks the frozen permission, retry, Attempt, usage, and
terminal facts before sending the next frame. The installed runner emits
`job/done` after session flush, handle disposal, and persistence retirement.
The final recursive inventory requires exactly two observer sidecars and two
session artifacts, followed by `shutdown`, `bye`, and exit 0.

`--verify-existing` requires the original artifact tree at its original absolute
location because the raw transport and session headers retain those paths. The
locally generated `batch-preflight-transcript.jsonl` normalizes paths and is
saved in the output directory. The JSON receipt printed to standard output
names that transcript and records its derived facts.

The one-shot fixture check requires one logical Call,
two observable Attempts, one durable DSH retry event, a `router-standard`
session header, the frozen five-tool first-Request surface, exact terminal
provider usage, and one decoded packed chunk row. The offline Qwen receipt
accepts valid raw or packed chunk rows and verifies the
session/observer/Generation Probe join.

Run the local preflight with the Python environment that already contains the
GPU stack and the exact local snapshot:

```powershell
$env:DSH_QWEN_PYTHON = 'C:\path\to\gpu-venv\Scripts\python.exe'
$env:DSH_QWEN_SNAPSHOT = 'C:\path\to\Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28'
node .\scripts\collection\dsh-headless\run-qwen-preflight.mjs C:\short\qwen-dsh-preflight 4
```

The launcher targets the current Windows installation. Set `DSH_PACKAGE_ROOT`
when DSH is installed outside the default global npm location.

The local preflight uses a 4-byte checkpoint cadence to exercise the online
Probe on a short deterministic response. Formal collection retains the frozen
128-byte cadence. The Qwen receipt requires two logical Calls and two Attempts with zero retry:
the first Call emits the only `pwsh` tool call, the tool result contains the
fixed marker, and the second Call reaches `stop`. It cross-checks raw DSH usage,
observer usage, exact tokenizer accounting, NF4 runtime attestation, router
promotion, online checkpoint replay, and the Generation Probe sidecar. Probe
rows join one-to-one by `session_id`, `attempt_id`, `checkpoint_ordinal`, and
`observer_committed_tokens`; each row also carries its audited source-output
Token ordinal. The external provider request count is 0.

Recheck an existing preflight capture without loading the model:

```powershell
node .\scripts\collection\dsh-headless\run-qwen-preflight.mjs --verify-existing C:\short\qwen-dsh-preflight 4
```

The preflights print JSON receipts with installed component versions, local
artifact paths, and verification results. Save standard output to retain a
receipt. The local Qwen preflight's fixed `Write-Output` command has no provider side
effect. Commercial actor collection uses the existing DSH-managed credential.
The capture validates the configured and
effective provider, model, reasoning mode and route for every Request.
DSH rc.2 successful streams do not expose the provider response model or the
provider request identity. The DSH manifest marks both fields unavailable for
this runtime, and the capture leaves them unfilled. Preflight receipts apply
to their recorded runtime and condition; later model or endpoint changes
require their own validation.
