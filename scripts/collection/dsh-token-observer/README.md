# DSH Token observer

This package observes ordinary DeepSeek Harness Agent-loop requests at
`llm/stream`. It writes a separate JSONL sidecar for each DSH session and leaves
the native session event log unchanged.

The schema-v2 sidecar records one natural `request_id` per logical Call, every
provider Attempt, visible-text checkpoints, terminal usage, and retry ownership. Reasoning deltas and tool-call
arguments do not enter the visible-output counter. The observer unit is the
UTF-8 byte length of a well-formed visible text stream; a trailing UTF-16 high
surrogate is carried into the next delta and isolated surrogates become the
Unicode replacement character.

A chunk becomes an observer fact after the downstream consumer resumes the
stream. Cancellation therefore excludes the last delivered, unacknowledged
chunk and closes the Attempt with `consumer-cancelled`. A thrown consumer error
uses `consumer-exception`; an upstream iterator failure uses
`stream-exception`. These terminals set `observer_finalized` to `false`.

Version 0.4.0 keeps one exclusive append descriptor per session and
synchronously appends each complete JSONL record. A record-write failure
poisons that session writer: later writes and its flush surface the stored
error. `session/flush` is an `fsync` barrier for dirty observer data.
`session/disposed` runs the barrier and closes the descriptor, while plugin
teardown closes any remaining session writers.

The Agent-loop gate is identity-sensitive: DSH stores marked requests in a
process-local `WeakSet`. This plugin resolves `@deepseek-ai/dsh-llm` through
DSH's own `profiles/node_modules` fallback, which points to the running DSH
installation. The observer and Agent loop consequently share one physical
module instance when the package is linked by `dsh-super-injector`.

The default output directory is
`%USERPROFILE%\.dsh\token-prediction\sidecars`. The default checkpoint interval
is 128 UTF-8 bytes. A DSH plugin configuration can set `outputDir` and
`checkpointIntervalBytes`.

## Opt-in request content ledger

Set `requestContentLedger: true` to write an additional
`<outputDir>/request-content/<encoded-session-id>.jsonl`. `requestContentOutputDir`
overrides that directory. The same session flush/disposal barriers cover both
sidecars. Default collection leaves this feature disabled.

Each `request/content` record contains the final immutable DSH client request
seen at `llm/stream`, its canonical JSON SHA-256 hash, logical Call and Attempt
IDs, the current native sequence cutoff, and a fixed initial snapshot cutoff.
The first Attempt sets `snapshot: true`; retries retain the snapshot ID and
request hash while receiving separate Attempt IDs. The adapter's private wire
serialization and hidden transport retries are outside this capture boundary;
`provider_wire_attempt` is `unknown`.

`parts` covers string leaves of message content, the system string, and each
tool schema serialized as canonical JSON. Message role/source metadata and
other request options remain in `request`. Each part stores a source identity,
source text, source UTF-8 byte length, and half-open retained byte ranges.
Source identity distinguishes separate messages/tool calls with equal text.
The byte encoding replaces isolated UTF-16 surrogates with U+FFFD. These byte
units are separate from provider usage tokens. `request_utf8_bytes` measures
the complete canonical JSON request, including JSON framing.

The headless runner exposes two optional job fields:

- `tool_result_max_utf8_bytes`: a nonnegative integer caps the text/reasoning
  prefix of each newly produced tool result. The `tools/post-execute` hook
  changes the content placed in message history. This condition studies tool
  result length. It does not expire old retained content.
- `tool_result_history_keep_recent`: a nonnegative integer retains the newest
  K tool results and clears the content of older results at `agent/pre-step`.
  This condition changes reuse of already-visible content in future requests.
  K=2 provides the `recent2` condition. K=0 clears all past results.

The corresponding single-run config fields are `toolResultMaxUtf8Bytes` and
`toolResultHistoryKeepRecent`. Omission or null job values disable each policy.
History expiry uses DSH's append-only surface replacement contract: the log
keeps the original `tool/result`, a `compaction/prune` record uses the harness
token meter's estimate, and a replacement cites the original native sequence.
The ledger records removal sequence IDs and `removed_parts` with the original
content IDs/text and empty retained ranges. `parts` always describes the actual
current request. A missing source, non-JSON value, altered retry request, or
later policy/finalizer mutation produces `status: missing` and explicit
`missing_fields`; consumers must withhold audited reuse labels for these rows.

The prefix policy accepts text/reasoning tool content. Unsupported rich results
raise a policy error. DSH can contain tool-pipeline errors as tool error results;
the ledger flags missing prefix provenance when such a result reaches a request.
The history window supports any tool result content because it expires the
entire result body. All retained and removed source text is stored locally in
the opt-in ledger.

The installed `dsh-super-injector` can load this directory directly during
development. A persistent DSH installation can add the package directory to
the chosen profile. No model credential is read or copied by this package.

Run the observer and installed-DSH integration tests with:

```powershell
node --test .\test\*.test.mjs
```

Regenerate the deterministic integration fixture with an installed DSH package:

```powershell
node .\scripts\capture-fixture.mjs [output-directory] [dsh-package-root]
```

The default output is `dsh-capture` in the system temporary directory.

The script runs the real DSH Agent loop, retry path, tool runtime, and JSONL
persistence with a local deterministic adapter. It makes no network or
commercial-model call.

The companion Python reader accepts a capture directory containing the native
`session.jsonl` or `session.jsonl.zstd` plus `observer.jsonl`. Copy the matching
sidecar into that capture directory before dataset normalization. During batch
recovery, a torn final JSONL record quarantines the recovered artifact set in
the affected capture and suppresses replay. The durable prefix remains intact,
and recovery does not synthesize a missing Attempt terminal.
