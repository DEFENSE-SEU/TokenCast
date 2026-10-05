import assert from "node:assert/strict"
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import test from "node:test"
import { tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

import { parseObserverJsonl, verifyPreflight } from "../preflight-contract.mjs"
import { providerAccounting } from "../provider-accounting.mjs"
import { verifyInstalledRuntime } from "../host.mjs"

const here = dirname(fileURLToPath(import.meta.url))
const SESSION_TEXT = [
  {
    type: "session",
    version: 0,
    id: "session-fixture-preflight",
    cwd: "fixture-workspace",
    agentPreset: "router-standard",
  },
  { type: "permission/preset", seq: 0, data: { preset: "unrestricted" } },
  { type: "sandbox/mode", seq: 1, data: { mode: "danger-full-access" } },
  { type: "approval/policy", seq: 2, data: { policy: "never" } },
  {
    type: "text-chunks",
    seq0: 3,
    time0: 1,
    data: { turn: 1, step: 1, index: 0, dt: [1, 1], texts: ["a", "b", "c"] },
  },
  { type: "request/header", seq: 6 },
  { type: "llm/retry", seq: 7 },
].map((row) => JSON.stringify(row)).join("\n") + "\n"

function decodeStorageRecord(row) {
  if (row.type !== "text-chunks") return [row]
  return row.data.texts.map((text, index) => ({
    type: "assistant/chunk",
    seq: row.seq0 + index,
    time: row.time0 + index,
    data: {
      turn: row.data.turn,
      step: row.data.step,
      chunk: { type: "text-delta", index: row.data.index, text },
    },
  }))
}

function rows() {
  const sessionId = "session-fixture-preflight"
  return [
    {
      schema_version: 2,
      record_type: "observer/session",
      plugin_version: "0.4.0",
      dsh_llm_compatibility: ">=0.1.1-rc.2 <0.2.0",
      observer_durability_contract_id: "observer-jsonl-fsync-on-session-flush-v1",
    },
    {
      schema_version: 2,
      record_type: "attempt/start",
      logical_call_id: "call-1",
      request_id: "call-1:request",
      attempt_id: "attempt-1",
      attempt_ordinal: 1,
      retry_of_attempt_id: null,
      provider: "dsh-fixture",
      model: "dsh-fixture-model",
      reasoning_effort: "off",
      request_shape: { tool_count: 5 },
    },
    {
      schema_version: 2,
      record_type: "attempt/end",
      attempt_id: "attempt-1",
      finish_reason: "error",
      failure_code: "DSH_FIXTURE_RETRY",
      observer_finalized: false,
    },
    {
      schema_version: 2,
      record_type: "attempt/start",
      logical_call_id: "call-1",
      request_id: "call-1:request",
      attempt_id: "attempt-2",
      attempt_ordinal: 2,
      retry_of_attempt_id: "attempt-1",
      provider: "dsh-fixture",
      model: "dsh-fixture-model",
      reasoning_effort: "off",
      request_shape: { tool_count: 5 },
    },
    {
      schema_version: 2,
      record_type: "attempt/end",
      attempt_id: "attempt-2",
      finish_reason: "stop",
      failure_code: null,
      observer_finalized: true,
      usage: { inputTokens: 20, outputTokens: 4 },
    },
  ].map((row) => ({ ...row, session_id: sessionId }))
}

test("preflight contract accepts one retry inside one logical Call", () => {
  const receipt = verifyPreflight({
    observerRows: rows(),
    sessionText: SESSION_TEXT,
    stdout: "fixture complete\n",
    decodeStorageRecord,
    requirePackedSession: true,
  })
  assert.deepEqual(
    { logicalCallCount: receipt.logicalCallCount, attemptCount: receipt.attemptCount },
    { logicalCallCount: 1, attemptCount: 2 },
  )
})

test("preflight contract reports workspace identity drift precisely", () => {
  assert.throws(
    () => verifyPreflight({
      observerRows: rows(),
      sessionText: SESSION_TEXT,
      stdout: "fixture complete\n",
      decodeStorageRecord,
      requirePackedSession: true,
      expectedWorkspace: "different-workspace",
    }),
    /session workspace differs from the requested workspace/,
  )
})

test("preflight contract rejects observer runtime drift", () => {
  const observerRows = rows()
  observerRows[0].plugin_version = "0.5.0"
  assert.throws(
    () => verifyPreflight({
      observerRows,
      sessionText: SESSION_TEXT,
      stdout: "fixture complete\n",
      decodeStorageRecord,
      requirePackedSession: true,
    }),
    /observer plugin compatibility drifted/,
  )
})

test("deterministic observer parser requires strict JSONL", () => {
  const text = `${rows().map((row) => JSON.stringify(row)).join("\n")}\n`
  assert.equal(parseObserverJsonl(text).length, rows().length)
  assert.throws(() => parseObserverJsonl(text.slice(0, -1)), /must end with LF/)
  assert.throws(() => parseObserverJsonl(`${text}\n`), /blank row/)
})

test("installed runtime rejects a DSH home Cordis overlay", () => {
  const dshHome = mkdtempSync(join(tmpdir(), "dsh-home-overlay-"))
  try {
    writeFileSync(join(dshHome, "cordis.patch.yml"), "[]\n")
    assert.throws(
      () => verifyInstalledRuntime({ dshRoot: dshHome, dshHome, profile: "dsh-headless" }),
      /home cordis\.patch\.yml overlay is forbidden/i,
    )
  } finally {
    rmSync(dshHome, { recursive: true, force: true })
  }
})

test("installed runtime requires the DSH profile identity", () => {
  assert.throws(
    () => verifyInstalledRuntime({ dshRoot: "unused", dshHome: "unused", profile: "headless" }),
    /requires the frozen dsh-headless profile/i,
  )
})

test("installed runtime rejects a nonempty profile patch", () => {
  const dshHome = mkdtempSync(join(tmpdir(), "dsh-profile-overlay-"))
  try {
    const profileRoot = join(dshHome, "profiles", "dsh-headless")
    mkdirSync(profileRoot, { recursive: true })
    writeFileSync(join(profileRoot, "cordis.yml"), "[]\n")
    writeFileSync(join(profileRoot, "cordis.patch.yml"), "- id: injected\n")
    assert.throws(
      () => verifyInstalledRuntime({ dshRoot: dshHome, dshHome, profile: "dsh-headless" }),
      /profile cordis\.patch\.yml must remain an empty list/i,
    )
  } finally {
    rmSync(dshHome, { recursive: true, force: true })
  }
})

test("bundle model compatibility selects the correct thinking protocol", () => {
  const patch = readFileSync(join(here, "../cordis.patch.yml"), "utf8")
  const evaluate = (field, model, effort = "off") => {
    const line = patch.split(/\r?\n/).find((line) => line.trim().startsWith(`${field}: !!js`))
    const expression = JSON.parse(line.split("!!js ")[1])
    return new Function("process", `return (${expression})`)({ env: { DSH_MODEL: model, DSH_REASONING_EFFORT: effort } })
  }
  for (const model of ["google/gemini-3.5-flash", "gemini-2.5-flash"]) {
    assert.equal(evaluate("thinkingFormat", model), "openai")
    assert.equal(evaluate("supportsReasoningEffort", model), true)
    assert.equal(evaluate("reasoningEfforts", model).low, "low")
    assert.equal(evaluate("reasoningEfforts", model).off, "none")
  }
  for (const model of ["moonshotai/kimi-k2.5", "kimi-k2.5"]) {
    assert.equal(evaluate("thinkingFormat", model), "deepseek")
    assert.equal(evaluate("supportsReasoningEffort", model), false)
    assert.equal(evaluate("requiresReasoningContentOnAssistantMessages", model), true)
    assert.deepEqual(evaluate("reasoningEfforts", model), { off: "none", high: "high" })
  }
  for (const model of ["qwen3-235b-a22b-instruct-2507", "openai/gpt-5.4"]) {
    assert.equal(evaluate("thinkingFormat", model), "qwen")
    assert.deepEqual(evaluate("reasoningEfforts", model), { off: null, low: "low" })
  }
  assert.equal(evaluate("thinkingFormat", "openai/gpt-5.4", "low"), "openai")
  assert.equal(evaluate("maxTokensField", "openai/gpt-5.4", "low"), "max_completion_tokens")
  assert.equal(evaluate("maxTokensField", "openai/gpt-5.4", "off"), "max_tokens")
  assert.equal(evaluate("maxTokensField", "qwen3-235b-a22b-instruct-2507"), "max_tokens")
})

test("bundle patch mounts the preset-aware runner after the observer", () => {
  const patch = readFileSync(join(here, "../cordis.patch.yml"), "utf8")
  assert.match(patch, /id: session-title-llm\s+disabled: true/)
  assert.match(patch, /id: headless-runner\s+disabled: true/)
  assert.match(patch, /id: token-prediction-dsh-observer/)
  assert.match(patch, /id: dsh-agent-presets/)
  assert.match(patch, /id: tools\s+config:\s+mode: native/)
  assert.match(
    patch,
    /id: sandbox-policy[\s\S]*?mode: danger-full-access[\s\S]*?workspaceRoot: !!js process\.cwd\(\)/,
  )
  assert.match(patch, /id: approval[\s\S]*?policy: never/)
  assert.match(
    patch,
    /id: permission[\s\S]*?defaultPreset: unrestricted[\s\S]*?sandbox: danger-full-access[\s\S]*?approval: never/,
  )
  assert.match(patch, /compression: zstd/)
  assert.match(patch, /packChunks: true/)
  assert.match(patch, /writeBatchMaxDelayMs: 200/)
  assert.match(patch, /id: agent-loop[\s\S]*?maxParallelToolCalls: 10/)
  assert.match(patch, /name: '@deepseek-ai\/dsh-agent-presets'/)
  assert.match(
    patch,
    /id: dsh-agent-presets[\s\S]*?roots:[\s\S]*?path: !!js process\.env\.DSH_PRESET_ROOT[\s\S]*?trust: system[\s\S]*?includeUserRoot: false/,
  )
  assert.match(patch, /name: '@token-prediction\/dsh-headless\/runner'/)
  assert.match(patch, /agentPreset: !!js process\.env\.DSH_AGENT_PRESET/)
  assert.match(
    patch,
    /checkpointIntervalBytes: !!js Number\(process\.env\.DSH_CHECKPOINT_INTERVAL_BYTES \|\| '128'\)/,
  )
  assert.match(
    patch,
    /countToolArguments: !!js process\.env\.DSH_COUNT_TOOL_ARGUMENTS === '1'/,
  )
  assert.match(
    patch,
    /id: llm-deepseek[\s\S]*?process\.env\.DSH_PROVIDER === 'openai-codex'[\s\S]*?apiKeyEnv: !!js process\.env\.DSH_API_KEY_ENV \|\| 'DEEPSEEK_API_KEY'/,
  )
  assert.match(
    patch,
    /baseURL: !!js process\.env\.DSH_BASE_URL \|\| 'https:\/\/api\.deepseek\.com'/,
  )
  assert.match(patch, /id: llm-deepseek[\s\S]*?thinking: disabled[\s\S]*?reasoningEffort: off/)
  assert.match(
    patch,
    /models:[\s\S]*?id: deepseek-v4-flash[\s\S]*?id: deepseek-v4-pro[\s\S]*?streamIdleTimeoutMs: 300000/,
  )
  const provider = patch.slice(patch.indexOf('id: llm-deepseek'))
  const ceilings = [...provider.matchAll(/maxTokens: (\d+)/g)].map((m) => Number(m[1]))
  assert.ok(ceilings.length > 1, 'provider entry should list per-model ceilings')
  assert.ok(
    ceilings.every((value) => value === ceilings[0]),
    `per-model ceilings disagree: ${ceilings.join(', ')}`,
  )
  assert.match(
    patch,
    /id: llm-deepseek[\s\S]*?retryPolicy:[\s\S]*?mode: normal[\s\S]*?maxRetries: 1[\s\S]*?EMPTY_RESPONSE[\s\S]*?RATE_LIMIT[\s\S]*?SERVER[\s\S]*?TIMEOUT[\s\S]*?TRANSPORT[\s\S]*?initialDelayMs: 500[\s\S]*?maxDelayMs: 10000[\s\S]*?jitterRatio: 0/,
  )
  assert.match(
    patch,
    /id: dsh-subscription-adapter[\s\S]*?name: '@token-prediction\/dsh-headless\/subscription-adapter'[\s\S]*?process\.env\.DSH_PROVIDER !== 'openai-codex'[\s\S]*?DSH_OPENAI_AUTH_FILE[\s\S]*?provider: openai-codex[\s\S]*?gpt-5\.4[\s\S]*?maxRetries: !!js Number\(process\.env\.DSH_MAX_RETRIES \|\| '1'\)/,
  )
  assert.match(
    patch,
    /id: llm-pi-ai[\s\S]*?process\.env\.DSH_PROVIDER !== 'qwen-compatible'[\s\S]*?qwen-compatible:[\s\S]*?apiKeyEnv: !!js process\.env\.DSH_API_KEY_ENV \|\| 'QWEN_API_KEY'[\s\S]*?api: openai-completions[\s\S]*?baseURL: !!js process\.env\.DSH_BASE_URL[\s\S]*?timeoutMs: 300000[\s\S]*?streamIdleTimeoutMs: 300000[\s\S]*?supportsUsageInStreaming: true[\s\S]*?id: !!js process\.env\.DSH_MODEL[\s\S]*?reasoningEfforts: !!js[\s\S]*?supportsDeveloperRole: false[\s\S]*?thinkingFormat: !!js[\s\S]*?retryPolicy:[\s\S]*?maxRetries: !!js Number\(process\.env\.DSH_MAX_RETRIES \|\| '1'\)/,
  )
  assert.match(
    patch,
    /id: llm-deepseek[\s\S]*?process\.env\.DSH_PROVIDER === 'qwen-compatible'/,
  )
  assert.match(
    patch,
    /id: dsh-aggregator-adapter[\s\S]*?name: '@token-prediction\/dsh-headless\/aggregator-adapter'[\s\S]*?process\.env\.DSH_PROVIDER !== 'qwen-compatible'/,
  )
  for (const id of [
    "settings",
    "session-telemetry-otel",
    "compaction-basic",
    "command-compact",
    "tool-result-pruner",
    "web",
    "web-search-deepseek",
    "tool-web",
    "agent-instructions",
    "tool-bash",
    "tool-pwsh",
    "tool-jobs",
    "tool-fs",
    "tool-fs-search",
    "skill-filesystem",
    "tool-skill",
    "plan-mode",
    "tool-goal",
    "tool-str-replace-editor",
    "tool-todo",
    "subagent",
    "workflow-worker-thread",
    "tool-workflow",
  ]) {
    assert.match(patch, new RegExp(`(?:^|\\n)- id: ${id}\\r?\\n  disabled: true`))
  }
  assert.match(patch, /maxLogicalCalls: !!js "process\.env\.DSH_MAX_LOGICAL_CALLS === 'unbounded' \? undefined : Number/)
  assert.match(
    patch,
    /maxProviderAccountedTokens: !!js "process\.env\.DSH_MAX_PROVIDER_ACCOUNTED_TOKENS === 'unbounded' \? undefined : Number/,
  )
  assert.match(patch, /sessionExportPath: !!js process\.env\.DSH_SESSION_EXPORT_PATH/)
  assert.match(patch, /dryRun: !!js process\.env\.DSH_DRY_RUN === '1'/)
})

test("frozen preset owns the exact DSH actor plane without auxiliary model calls", () => {
  const packageJson = JSON.parse(readFileSync(join(here, "../package.json"), "utf8"))
  assert.ok(packageJson.files.includes("preset-root/"))

  const preset = readFileSync(
    join(here, "../preset-root/router-standard/agent.cordis.yml"),
    "utf8",
  )
  for (const id of [
    "persona",
    "agent-instructions",
    "router-bootstrap",
    "tool-pwsh",
    "tool-fs",
    "tool-fs-search",
    "tool-jobs",
    "skill-filesystem",
    "tool-skill",
    "tool-str-replace-editor",
    "tool-todo",
  ]) {
    assert.match(preset, new RegExp(`(?:^|\\n)\\s*- id: ${id}(?:\\r?\\n|$)`))
  }
  for (const id of [
    "tool-bash",
    "tool-goal",
    "planning",
    "plan-mode",
    "tool-ask-user",
    "tool-web",
    "delegation",
    "tool-workflow",
    "compaction-basic",
    "command-compact",
    "tool-result-pruner",
  ]) {
    assert.doesNotMatch(preset, new RegExp(`(?:^|\\n)\\s*- id: ${id}(?:\\r?\\n|$)`))
  }
  assert.match(
    preset,
    /skill-filesystem[\s\S]*?includeDefaultRoots: true[\s\S]*?runtime-home[\s\S]*?runtime-agents[\s\S]*?runtime-bundled-skills[\s\S]*?watch: false/,
  )
  assert.match(
    preset,
    /agent-instructions[\s\S]*?dshHome: !!js process\.env\.DSH_PRESET_ROOT \+ '\/runtime-home'[\s\S]*?maxBytes: 65536/,
  )

  const rootFiles = [
    "agent.cordis.yml",
    "preset.yml",
    "router-bootstrap.mjs",
    "router-core.mjs",
  ]
  const rootText = rootFiles.map((name) => readFileSync(
    join(here, "../preset-root/router-standard", name),
    "utf8",
  )).join("\n")
  const bannedTerms = ["co" + "dex", "ba" + "gen", "sp" + "end", "sha" + "256"]
  assert.doesNotMatch(rootText, new RegExp(`\\b(?:${bannedTerms.join("|")})\\b`, "i"))

  const router = readFileSync(
    join(here, "../preset-root/router-standard/router-bootstrap.mjs"),
    "utf8",
  )
  assert.doesNotMatch(router, /dev_(?:router|mode)|ctx\.llm|toJsonSchema|currentAgent/)
  assert.match(router, /classifyTask/)
  assert.match(router, /agent\/inbox\/claimed/)
  assert.match(router, /agent\/pre-step/)

  const fixturePreflight = readFileSync(join(here, "../run-preflight.mjs"), "utf8")
  assert.match(fixturePreflight, /DSH_PRESET_ROOT: join\(here, "preset-root"\)/)
  assert.match(fixturePreflight, /DSH_EXCLUSIVE_PROVIDER: "dsh-fixture"/)
  assert.match(fixturePreflight, /DSH_LOCAL_QWEN: "0"/)
  const qwenPreflight = readFileSync(join(here, "../run-qwen-preflight.mjs"), "utf8")
  assert.match(qwenPreflight, /DSH_PRESET_ROOT: join\(here, "preset-root"\)/)
})

test("package publishes every exported adapter module", () => {
  const packageJson = JSON.parse(readFileSync(join(here, "../package.json"), "utf8"))
  for (const target of Object.values(packageJson.exports)) {
    assert.ok(packageJson.files.includes(target.replace(/^\.\//, "")), target)
  }
})

test("provider accounting is disjoint and detects missing terminal usage", () => {
  const complete = providerAccounting([
    {
      type: "assistant/chunk",
      data: {
        chunk: {
          type: "usage",
          usage: {
            inputTokens: 10,
            outputTokens: 4,
            cacheReadTokens: 3,
            cacheWriteTokens: 2,
            reasoningTokens: 1,
          },
        },
      },
    },
    { type: "assistant/chunk", data: { chunk: { type: "finish", reason: { kind: "stop" } } } },
  ])
  assert.deepEqual(complete, { tokens: 19, terminalWithoutUsage: false })

  const missing = providerAccounting([
    { type: "assistant/chunk", data: { chunk: { type: "finish", reason: { kind: "error" } } } },
  ])
  assert.deepEqual(missing, { tokens: 0, terminalWithoutUsage: true })
})

test("archived receipt records the free installed-DSH preflight", (t) => {
  const path = join(here, "../preflight-receipt.json")
  if (!existsSync(path)) {
    t.skip("The installed-DSH preflight receipt is excluded from the source release")
    return
  }
  const receipt = JSON.parse(readFileSync(path, "utf8"))
  assert.equal(receipt.status, "passed")
  assert.equal(receipt.external_provider_request_count, 0)
  assert.equal(receipt.logical_call_count, 1)
  assert.equal(receipt.attempt_count, 2)
  assert.equal(receipt.retry_count, 1)
  assert.equal(receipt.session_packed_row_count, 1)
  assert.equal(receipt.components.dsh_cli, "0.1.1-rc.2")
  assert.equal(receipt.components.dsh_agent_presets, "0.1.1-rc.2")
  assert.equal(receipt.components.dsh_agent_loop, "0.1.1-rc.2")
  assert.equal(receipt.components.dsh_agent, "0.1.1-rc.2")
  assert.equal(receipt.components.dsh_llm_deepseek, "0.1.1-rc.2")
  assert.equal(receipt.components.dsh_session_checkpoint_policy, "0.1.1-rc.2")
  assert.equal(receipt.components.dsh_session_persistence, "0.1.1-rc.2")
  assert.equal(receipt.components.dsh_token_meter, "0.1.1-rc.2")
  assert.equal(receipt.components.dsh_tools, "0.1.1-rc.2")
  assert.equal(receipt.components.token_prediction_observer, "0.4.0")
  assert.equal(receipt.components.token_prediction_dsh_headless, "0.5.0")
  assert.equal(receipt.runtime.headless_integration_version, "0.5.0")
  assert.equal(receipt.runtime.observer_plugin_version, "0.4.0")
  assert.equal(receipt.runtime.batch_protocol_version, 2)
  assert.equal(
    receipt.runtime.observer_durability_contract_id,
    "observer-jsonl-fsync-on-session-flush-v1",
  )
  assert.deepEqual(receipt.runtime.session_persistence, {
    compression: "zstd",
    pack_chunks: true,
    physical_artifact: "session.jsonl.zstd",
    decoded_preflight_export: "session.jsonl",
    one_live_writer_per_session: true,
    packed_row_count: 1,
  })
  assert.equal(receipt.runtime.max_parallel_tool_calls, 10)
  assert.equal(receipt.runtime.agent_handle_disposed, true)
  assert.equal(receipt.agent_preset, "router-standard")
  assert.equal(receipt.workspace_identity_verified, true)
  assert.equal(receipt.permission_preset, "unrestricted")
  assert.equal(receipt.sandbox_mode, "danger-full-access")
  assert.equal(receipt.approval_policy, "never")
  assert.equal(receipt.first_request_tool_count, 5)
  assert.deepEqual(receipt.effective_condition, {
    provider: "dsh-fixture",
    model: "dsh-fixture-model",
    reasoning_effort: "off",
    temperature: 0,
    max_tokens: 64,
    stop: [],
  })
})

test("archived Qwen receipt records the offline installed-profile preflight", (t) => {
  const path = join(here, "../qwen-preflight-receipt.json")
  if (!existsSync(path)) {
    t.skip("The installed-profile Qwen preflight receipt is excluded from the source release")
    return
  }
  const receipt = JSON.parse(readFileSync(path, "utf8"))
  assert.equal(receipt.status, "passed")
  assert.equal(receipt.profile, "dsh-headless")
  assert.equal(receipt.components.dsh_agent_presets, "0.1.1-rc.2")
  assert.equal(receipt.components.dsh_agent, "0.1.1-rc.2")
  assert.equal(receipt.components.dsh_llm_deepseek, "0.1.1-rc.2")
  assert.equal(receipt.components.dsh_llm_retry, "0.1.1-rc.2")
  assert.equal(receipt.components.dsh_session_checkpoint_policy, "0.1.1-rc.2")
  assert.equal(receipt.components.dsh_session_persistence, "0.1.1-rc.2")
  assert.equal(receipt.components.dsh_tools, "0.1.1-rc.2")
  assert.equal(receipt.components.router_standard, "0.3.0")
  assert.equal(receipt.components.token_prediction_dsh_headless, "0.5.0")
  assert.equal(receipt.runtime.dsh_cli_version, "0.1.1-rc.2")
  assert.equal(receipt.runtime.dsh_core_version, "0.1.1-rc.2")
  assert.equal(receipt.runtime.headless_integration_version, "0.5.0")
  assert.equal(receipt.runtime.observer_plugin_version, "0.4.0")
  assert.equal(receipt.runtime.batch_protocol_version, 2)
  assert.equal(
    receipt.runtime.observer_durability_contract_id,
    "observer-jsonl-fsync-on-session-flush-v1",
  )
  assert.equal(receipt.runtime.session_probe_join_verified, true)
  assert.equal(receipt.verified_scope, "local_adapter_transport_tool_accounting")
  assert.equal(receipt.external_provider_request_count, 0)
  assert.deepEqual(receipt.external_provider_evidence, {
    local_request_header_count: 2,
    local_attempt_count: 2,
    session_title_source: "fallback",
  })
  assert.equal(receipt.logical_call_count, 2)
  assert.equal(receipt.attempt_count, 2)
  assert.equal(receipt.retry_count, 0)
  assert.equal(receipt.runtime.session_persistence.packed_row_count, 0)
  assert.equal(receipt.session_packed_row_count, 0)
  assert.equal(receipt.workspace_identity_verified, true)
  assert.equal(receipt.permission_preset, "unrestricted")
  assert.equal(receipt.sandbox_mode, "danger-full-access")
  assert.equal(receipt.approval_policy, "never")
  assert.equal(receipt.tool_name, "pwsh")
  assert.equal(receipt.final_response_contract, "ends_with_unique_marker")
  assert.equal(receipt.terminal_instruction_compliant, true)
  assert.equal(
    receipt.observed_final_response_contract,
    "unique_marker_only",
  )
  assert.equal(receipt.promoted_request_tool_count, 12)
  assert.deepEqual(receipt.observer_checkpoint_contract, {
    evidence_scope: "preflight_probe",
    checkpoint_interval_bytes: 4,
    checkpoint_policy_id: "dsh-text-chunk-at-least-4-utf8-bytes-v1",
    frozen_collection_checkpoint_interval_bytes: 128,
    frozen_collection_checkpoint_policy_id: "dsh-text-chunk-at-least-128-utf8-bytes-v1",
    frozen_collection_policy_verified: false,
  })
  assert.deepEqual(receipt.observer_checkpoint_count, [0, 1])
  assert.deepEqual(receipt.acknowledged_delta_count, [15, 1])
  assert.deepEqual(receipt.generation_probe, {
    capture_revision_prefix: "local-qwen-generation-probe-v2:preflight",
    capture_revision_policy: "prefix-plus-session-id",
    checkpoint_count: 1,
    projection_policy_id: "rademacher-last-layer-v2",
    projection_dimension: 64,
    causal_alignment: "post_source_output_token_prefix_before_next_sample",
    source_output_tokens: [8],
  })
  const serialized = JSON.stringify(receipt)
  assert.equal(serialized.includes('"session_id":'), false)
  assert.doesNotMatch(serialized, /session-[0-9a-f]{8}-[0-9a-f-]{27}/i)
  assert.doesNotMatch(serialized, /[A-Z]:\\/)
})
