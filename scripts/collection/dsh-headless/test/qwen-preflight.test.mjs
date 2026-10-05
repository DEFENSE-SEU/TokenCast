import assert from "node:assert/strict"
import { existsSync, readFileSync, readdirSync } from "node:fs"
import test from "node:test"
import { join } from "node:path"

import {
  QWEN_PREFLIGHT_TASK,
  verifyGenerationProbePreflight,
  verifyQwenPreflight,
} from "../qwen-preflight-contract.mjs"
import {
  ALLOWED_TOOL_NAMES,
  TOOL_PROMPT_SECTION,
} from "../tool-policy.mjs"
import { importDshHostModule } from "../host.mjs"

const EVIDENCE_ROOT = process.env.DSH_QWEN_PREFLIGHT_EVIDENCE_DIR

function findFile(root, predicate) {
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    const path = join(root, entry.name)
    if (entry.isDirectory()) {
      const found = findFile(path, predicate)
      if (found) return found
    } else if (predicate(path, entry.name)) {
      return path
    }
  }
  return undefined
}

async function loadEvidence() {
  if (!EVIDENCE_ROOT || !existsSync(EVIDENCE_ROOT)) return undefined
  const sessionPath = join(EVIDENCE_ROOT, "session.jsonl")
  const observerPath = findFile(
    join(EVIDENCE_ROOT, "observer"),
    (_path, name) => name.endsWith(".jsonl"),
  )
  if (!existsSync(sessionPath) || !observerPath) return undefined
  const sessionText = readFileSync(sessionPath, "utf8")
  const observerText = readFileSync(observerPath, "utf8")
  const observerSession = JSON.parse(observerText.slice(0, observerText.indexOf("\n")))
  const { decodeStorageRecord } = await importDshHostModule("@deepseek-ai/dsh-session")
  return {
    sessionText,
    observerText,
    checkpointIntervalBytes: observerSession.checkpoint_interval_bytes,
    decodeStorageRecord,
    expectedWorkspace: join(EVIDENCE_ROOT, "workspace"),
    requirePackedSession: false,
    sessionId: JSON.parse(sessionText.slice(0, sessionText.indexOf("\n"))).id,
  }
}

function rewriteJsonl(text, mutate) {
  const rows = text.slice(0, -1).split("\n").map(JSON.parse)
  mutate(rows)
  return `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`
}

const INITIAL_TOOLS = ["edit", "pwsh", "read", "write"]
const PROMOTED_TOOLS = [...ALLOWED_TOOL_NAMES]
const CONFIG = {
  provider: "local-qwen",
  model: "Qwen/Qwen2.5-7B-Instruct",
  maxTokens: 192,
  temperature: 0,
  stop: [],
  reasoningEffort: "off",
}
const TOOL_ARGUMENTS = JSON.stringify({
  command: "Write-Output QWEN_DSH_PREFLIGHT_TOOL_OK",
  description: "Print the Qwen DSH preflight marker",
})
const PROVIDER_UNIT =
  "provider:dsh-outputTokens-v1:local-qwen:Qwen%2FQwen2.5-7B-Instruct"
const SYSTEM_PROMPT = `Fixture system prompt.\n\n${TOOL_PROMPT_SECTION.text}`
const PROMOTED_SYSTEM_PROMPT = `${SYSTEM_PROMPT}\n\nTrack every background job id you start.`

function replay(inputTokens, outputTokens, acknowledgedDeltaCount) {
  return {
    revision: "a09a35458c702b33eeacc393d103063234e8bc28",
    precision: "bitsandbytes-nf4-4bit",
    device: "cuda:0",
    modelClass: "Qwen2Model",
    isQuantized: true,
    quantType: "nf4",
    doubleQuant: true,
    computeDtype: "torch.bfloat16",
    tieWordEmbeddings: false,
    linear4bitModules: 196,
    lmHeadDtype: "torch.bfloat16",
    repetitionPenalty: 1.05,
    eosTokenIds: [151645, 151643],
    tokenizerInputTokens: inputTokens,
    tokenizerOutputTokens: outputTokens,
    acknowledgedDeltaCount,
  }
}

function makeFixtureEvidence({ toolOnlyFirstCall = false } = {}) {
  const sessionId = "fixture-qwen-preflight-session"
  const rows = [{
    type: "session",
    version: 0,
    id: sessionId,
    createdAt: 1,
    cwd: "fixture-workspace",
    delegationDepth: 0,
    agentPreset: "router-standard",
  }]
  const push = (type, data, time) => {
    const row = { type, seq: rows.length - 1, time, data }
    rows.push(row)
    return row
  }
  push("permission/preset", { preset: "unrestricted" }, 1)
  push("sandbox/mode", { mode: "danger-full-access" }, 1)
  push("approval/policy", { policy: "never" }, 1)
  const user = push("user/message", {
    content: [{ type: "text", text: QWEN_PREFLIGHT_TASK }],
    source: { kind: "user" },
    role: "user",
    id: "fixture-user-message",
  }, 2)
  push("session/title", {
    title: "Fixture fallback title",
    messageSeqs: [user.seq],
    source: { kind: "fallback" },
  }, 4)

  const callId = "fixture-pwsh-call"
  const firstText = toolOnlyFirstCall ? "" : "Planning the requested command.\n"
  const firstUsage = { inputTokens: 10, outputTokens: 5 }
  const firstReplay = replay(10, 5, 2)
  push("step/start", { turn: 1, step: 1 }, 5)
  push("request/header", {
    header: {
      config: CONFIG,
      system: SYSTEM_PROMPT,
      tools: INITIAL_TOOLS.map((name) => ({ name })),
    },
    reason: "initial",
  }, 6)
  push("request/context", {
    provider: CONFIG.provider,
    model: CONFIG.model,
    contextWindow: 32768,
  }, 7)
  const firstChunks = []
  if (!toolOnlyFirstCall) {
    firstChunks.push(
      push("assistant/chunk", { turn: 1, step: 1, chunk: { type: "block-start", index: 0, blockType: "text" } }, 10),
      push("assistant/chunk", { turn: 1, step: 1, chunk: { type: "text-delta", index: 0, text: firstText } }, 11),
      push("assistant/chunk", { turn: 1, step: 1, chunk: { type: "block-end", index: 0, block: { type: "text", text: firstText } } }, 12),
    )
  }
  const toolBlockIndex = toolOnlyFirstCall ? 0 : 1
  firstChunks.push(
    push("assistant/chunk", { turn: 1, step: 1, chunk: { type: "block-start", index: toolBlockIndex, blockType: "tool-call" } }, 13),
    push("assistant/chunk", { turn: 1, step: 1, chunk: { type: "tool-call-delta", index: toolBlockIndex, id: callId, name: "pwsh", argumentsDelta: TOOL_ARGUMENTS } }, 14),
    push("assistant/chunk", { turn: 1, step: 1, chunk: { type: "block-end", index: toolBlockIndex, block: { type: "tool-call", id: callId, name: "pwsh", arguments: TOOL_ARGUMENTS } } }, 15),
    push("assistant/chunk", { turn: 1, step: 1, chunk: { type: "usage", usage: firstUsage } }, 16),
    push("assistant/chunk", { turn: 1, step: 1, chunk: { type: "finish", reason: { kind: "tool-calls" }, replayState: { response: firstReplay } } }, 17),
  )
  push("assistant/message", {
    turn: 1,
    step: 1,
    message: {
      role: "assistant",
      content: [
        ...(!toolOnlyFirstCall ? [{ type: "text", text: firstText }] : []),
        { type: "tool-call", id: callId, name: "pwsh", arguments: TOOL_ARGUMENTS },
      ],
      source: {
        kind: "model",
        provider: CONFIG.provider,
        model: CONFIG.model,
        replayState: { response: firstReplay },
      },
      id: "fixture-first-assistant",
    },
    usage: firstUsage,
  }, 30).sourceEventSeqs = firstChunks.map((row) => row.seq)
  const toolCall = push("tool/call", {
    turn: 1,
    step: 1,
    callId,
    name: "pwsh",
    arguments: TOOL_ARGUMENTS,
  }, 31)
  const toolResult = push("tool/result", {
    turn: 1,
    step: 1,
    message: {
      source: { kind: "tool", callId },
      content: [{
        type: "tool-result",
        toolCallId: callId,
        content: [{ type: "text", text: "QWEN_DSH_PREFLIGHT_TOOL_OK\r\n" }],
        isError: false,
      }],
      role: "user",
      id: "fixture-tool-result",
    },
  }, 32)
  toolResult.sourceEventSeqs = [toolCall.seq]
  toolResult.surfaceOp = "append"
  push("step/end", { turn: 1, step: 1 }, 33)

  const terminalText = "The command succeeded.\n\nQWEN_DSH_PREFLIGHT_DONE"
  const secondUsage = { inputTokens: 20, outputTokens: 3 }
  const secondReplay = replay(20, 3, 1)
  push("step/start", { turn: 1, step: 2 }, 40)
  push("request/header", {
    header: {
      config: CONFIG,
      system: PROMOTED_SYSTEM_PROMPT,
      tools: PROMOTED_TOOLS.map((name) => ({ name })),
    },
    reason: "change",
  }, 41)
  const secondChunks = [
    push("assistant/chunk", { turn: 1, step: 2, chunk: { type: "block-start", index: 0, blockType: "text" } }, 42),
    push("assistant/chunk", { turn: 1, step: 2, chunk: { type: "text-delta", index: 0, text: terminalText } }, 43),
    push("assistant/chunk", { turn: 1, step: 2, chunk: { type: "block-end", index: 0, block: { type: "text", text: terminalText } } }, 44),
    push("assistant/chunk", { turn: 1, step: 2, chunk: { type: "usage", usage: secondUsage } }, 45),
    push("assistant/chunk", { turn: 1, step: 2, chunk: { type: "finish", reason: { kind: "stop" }, replayState: { response: secondReplay } } }, 46),
  ]
  push("assistant/message", {
    turn: 1,
    step: 2,
    message: {
      role: "assistant",
      content: [{ type: "text", text: terminalText }],
      source: {
        kind: "model",
        provider: CONFIG.provider,
        model: CONFIG.model,
        replayState: { response: secondReplay },
      },
      id: "fixture-terminal-assistant",
    },
    usage: secondUsage,
  }, 60).sourceEventSeqs = secondChunks.map((row) => row.seq)
  push("step/end", { turn: 1, step: 2 }, 61)
  push("turn/end", { turn: 1, reason: { kind: "completed" } }, 62)

  const observerHeader = {
    schema_version: 2,
    record_type: "observer/session",
    time: 8,
    session_id: sessionId,
    plugin_version: "0.4.0",
    dsh_llm_compatibility: ">=0.1.1-rc.2 <0.2.0",
    observer_durability_contract_id: "observer-jsonl-fsync-on-session-flush-v1",
    checkpoint_interval_bytes: 4,
    checkpoint_policy_id: "dsh-text-chunk-at-least-4-utf8-bytes-v1",
    observer_unit_contract_id: "observer:utf8-bytes-v1",
    observer_tokenizer_id: "utf8-byte-256",
    observer_normalization_id: "utf16-to-well-formed-v1",
    observer_carry_policy_id: "utf16-high-surrogate-carry-v1",
    observer_boundary_policy_id: "dsh-text-delta-chunk-v1",
  }
  const observerRows = [observerHeader]
  const addAttempt = ({ step, text, usage, finishReason, startTime, checkpointTime, endTime }) => {
    const logicalCallId = `${sessionId}:turn-1:step-${step}`
    const requestId = `${logicalCallId}:request`
    const attemptId = `${logicalCallId}:attempt-1`
    observerRows.push({
      schema_version: 2,
      record_type: "attempt/start",
      time: startTime,
      session_id: sessionId,
      turn: 1,
      step,
      logical_call_id: logicalCallId,
      request_id: requestId,
      attempt_id: attemptId,
      attempt_ordinal: 1,
      retry_of_attempt_id: null,
      provider: CONFIG.provider,
      model: CONFIG.model,
      reasoning_effort: "off",
      provider_token_unit_contract_id: PROVIDER_UNIT,
      request_shape: {
        message_count: step === 1 ? 1 : 3,
        tool_count: step === 1 ? 4 : PROMOTED_TOOLS.length,
        system_utf8_bytes: 24,
      },
    })
    if (text.length > 0) {
      observerRows.push({
        schema_version: 2,
        record_type: "attempt/checkpoint",
        time: checkpointTime,
        session_id: sessionId,
        turn: 1,
        step,
        logical_call_id: logicalCallId,
        attempt_id: attemptId,
        attempt_ordinal: 1,
        checkpoint_ordinal: 1,
        observer_committed_tokens: Buffer.byteLength(text, "utf8"),
        observer_unit_contract_id: "observer:utf8-bytes-v1",
        visible_text_delta: text,
      })
    }
    observerRows.push({
      schema_version: 2,
      record_type: "attempt/end",
      time: endTime,
      session_id: sessionId,
      turn: 1,
      step,
      logical_call_id: logicalCallId,
      attempt_id: attemptId,
      attempt_ordinal: 1,
      provider_token_unit_contract_id: PROVIDER_UNIT,
      finish_reason: finishReason,
      failure_code: null,
      usage: { ...usage, cacheReadTokens: null, cacheWriteTokens: null, reasoningTokens: null },
      observer_visible_output_tokens: Buffer.byteLength(text, "utf8"),
      observer_finalized: true,
      visible_text_tail: "",
    })
  }
  addAttempt({
    step: 1,
    text: firstText,
    usage: firstUsage,
    finishReason: "tool-calls",
    startTime: 9,
    checkpointTime: 20,
    endTime: 29,
  })
  addAttempt({
    step: 2,
    text: terminalText,
    usage: secondUsage,
    finishReason: "stop",
    startTime: 39,
    checkpointTime: 50,
    endTime: 59,
  })
  return {
    sessionText: `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`,
    observerText: `${observerRows.map((row) => JSON.stringify(row)).join("\n")}\n`,
    sessionId,
  }
}

const evidence = await loadEvidence()
const fixtureEvidence = makeFixtureEvidence()
const skipWithoutEvidence = { skip: evidence === undefined }

function fixtureProbeText(observerText = fixtureEvidence.observerText) {
  const observerRows = observerText.slice(0, -1).split("\n").map(JSON.parse)
  const sessionId = observerRows[0].session_id
  const rows = [{
    schema_version: 2,
    record_type: "generation_probe/session",
    contract_id: "dsh-generation-probe-v2",
    capture_revision: `local-qwen-generation-probe-v2:preflight:${sessionId}`,
    source_id: "dsh_session_probe_v2",
    session_id: sessionId,
    model_id: "Qwen/Qwen2.5-7B-Instruct",
    model_revision: "a09a35458c702b33eeacc393d103063234e8bc28",
    projection: {
      policy_id: "rademacher-last-layer-v2",
      policy_version: 2,
      source_layer: "last",
      dimension: 64,
      seed: 729,
      hidden_size: 3584,
      distribution: "rademacher",
      scaling: "inverse_sqrt_hidden_size",
      dtype: "float32",
      rng_algorithm: "torch-cpu-generator-v1",
      fit_scope: "fixed_without_dataset_fit",
    },
    causal_alignment: "post_source_output_token_prefix_before_next_sample",
  }]
  const sourceOutputByAttempt = new Map()
  for (const checkpoint of observerRows.filter(
    (row) => row.record_type === "attempt/checkpoint",
  )) {
    const sourceOutputTokens = (sourceOutputByAttempt.get(checkpoint.attempt_id) ?? 0) + 1
    sourceOutputByAttempt.set(checkpoint.attempt_id, sourceOutputTokens)
    rows.push({
      schema_version: 2,
      record_type: "generation_probe/checkpoint",
      session_id: sessionId,
      attempt_id: checkpoint.attempt_id,
      checkpoint_ordinal: checkpoint.checkpoint_ordinal,
      observer_committed_tokens: checkpoint.observer_committed_tokens,
      source_output_tokens: sourceOutputTokens,
      hidden_state_projection: Array.from({ length: 64 }, (_value, index) => index / 64),
    })
  }
  return `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`
}

test("exports the exact stable preflight task", () => {
  assert.equal(
    QWEN_PREFLIGHT_TASK,
    "Call the pwsh tool exactly once with command Write-Output QWEN_DSH_PREFLIGHT_TOOL_OK and description Print the Qwen DSH preflight marker. After the tool succeeds, reply with exactly QWEN_DSH_PREFLIGHT_DONE. Do not call any other tool.",
  )
})

test("strict JSONL rejects a missing final LF", () => {
  assert.throws(
    () => verifyQwenPreflight({ sessionText: "{}", observerText: "{}\n" }),
    /session JSONL must end with LF/,
  )
})

test("strict JSONL rejects blank rows", () => {
  assert.throws(
    () => verifyQwenPreflight({
      sessionText: '{"type":"session","id":"x","agentPreset":"router-standard"}\n\n',
      observerText: "{}\n",
    }),
    /session JSONL contains a blank line/,
  )
})

test("strict JSONL rejects a session seq gap", () => {
  assert.throws(
    () => verifyQwenPreflight({
      sessionText: '{"type":"session","version":0,"id":"x","cwd":"fixture-workspace","agentPreset":"router-standard"}\n{"type":"turn/end","seq":1}\n',
      observerText: "{}\n",
    }),
    /session seq must be continuous/,
  )
})

test("Generation Probe preflight enforces the complete sidecar contract", () => {
  const probeText = fixtureProbeText()
  assert.deepEqual(
    verifyGenerationProbePreflight({
      observerText: fixtureEvidence.observerText,
      probeText,
    }),
    {
      captureRevisionPrefix: "local-qwen-generation-probe-v2:preflight",
      captureRevisionPolicy: "prefix-plus-session-id",
      checkpointCount: 2,
      projectionPolicyId: "rademacher-last-layer-v2",
      projectionDimension: 64,
      causalAlignment: "post_source_output_token_prefix_before_next_sample",
      sourceOutputTokens: [1, 1],
    },
  )
  assert.throws(
    () => verifyGenerationProbePreflight({
      observerText: fixtureEvidence.observerText,
      probeText: probeText.slice(0, -1),
    }),
    /Generation Probe JSONL must end with LF/,
  )

  for (const [mutate, pattern] of [
    [(rows) => { rows[0].capture_revision += "-drift" }, /session contract drifted/],
    [(rows) => { rows[0].source_id = "other-source" }, /session contract drifted/],
    [(rows) => { rows[0].causal_alignment = "terminal-state" }, /session contract drifted/],
    [(rows) => { rows[1].label = 1 }, /checkpoint fields drifted/],
    [(rows) => { rows[1].source_output_tokens = 6 }, /exceeds terminal usage/],
  ]) {
    assert.throws(
      () => verifyGenerationProbePreflight({
        observerText: fixtureEvidence.observerText,
        probeText: rewriteJsonl(probeText, mutate),
      }),
      pattern,
    )
  }
})

test("Generation Probe preflight validates source positions in checkpoint order", () => {
  const observerText = rewriteJsonl(fixtureEvidence.observerText, (rows) => {
    const firstCheckpointIndex = rows.findIndex(
      (row) => row.record_type === "attempt/checkpoint" && row.step === 1,
    )
    const firstCheckpoint = rows[firstCheckpointIndex]
    rows.splice(firstCheckpointIndex + 1, 0, {
      ...firstCheckpoint,
      checkpoint_ordinal: 2,
      observer_committed_tokens: firstCheckpoint.observer_committed_tokens + 1,
    })
  })
  const probeText = rewriteJsonl(fixtureProbeText(), (rows) => {
    const firstCheckpointIndex = rows.findIndex(
      (row) => row.record_type === "generation_probe/checkpoint"
        && row.attempt_id.endsWith("step-1:attempt-1"),
    )
    const first = rows[firstCheckpointIndex]
    first.source_output_tokens = 5
    rows.splice(firstCheckpointIndex + 1, 0, {
      ...first,
      checkpoint_ordinal: 2,
      observer_committed_tokens: first.observer_committed_tokens + 1,
      source_output_tokens: 4,
    })
  })

  assert.throws(
    () => verifyGenerationProbePreflight({ observerText, probeText }),
    /source output position decreased within an Attempt/,
  )
})

test("Generation Probe preflight rejects sidecar checkpoint order drift", () => {
  const observerText = rewriteJsonl(fixtureEvidence.observerText, (rows) => {
    const firstCheckpointIndex = rows.findIndex(
      (row) => row.record_type === "attempt/checkpoint" && row.step === 1,
    )
    const firstCheckpoint = rows[firstCheckpointIndex]
    rows.splice(firstCheckpointIndex + 1, 0, {
      ...firstCheckpoint,
      checkpoint_ordinal: 2,
      observer_committed_tokens: firstCheckpoint.observer_committed_tokens + 1,
    })
  })
  const orderedProbeText = rewriteJsonl(fixtureProbeText(), (rows) => {
    const firstCheckpointIndex = rows.findIndex(
      (row) => row.record_type === "generation_probe/checkpoint"
        && row.attempt_id.endsWith("step-1:attempt-1"),
    )
    const firstCheckpoint = rows[firstCheckpointIndex]
    rows.splice(firstCheckpointIndex + 1, 0, {
      ...firstCheckpoint,
      checkpoint_ordinal: 2,
      observer_committed_tokens: firstCheckpoint.observer_committed_tokens + 1,
    })
  })
  assert.equal(
    verifyGenerationProbePreflight({ observerText, probeText: orderedProbeText }).checkpointCount,
    3,
  )

  const reorderedProbeText = rewriteJsonl(orderedProbeText, (rows) => {
    ;[rows[1], rows[2]] = [rows[2], rows[1]]
  })
  assert.throws(
    () => verifyGenerationProbePreflight({ observerText, probeText: reorderedProbeText }),
    /checkpoint order drifted within an Attempt/,
  )
})

test("Generation Probe preflight rejects global checkpoint order drift", () => {
  const probeText = rewriteJsonl(fixtureProbeText(), (rows) => {
    ;[rows[1], rows[2]] = [rows[2], rows[1]]
  })
  assert.throws(
    () => verifyGenerationProbePreflight({
      observerText: fixtureEvidence.observerText,
      probeText,
    }),
    /checkpoint order differs from observer checkpoints/,
  )
})

test("accepts a complete path-independent fixture success trajectory", () => {
  const receipt = verifyQwenPreflight(fixtureEvidence)
  assert.equal(receipt.status, "passed")
  assert.equal(receipt.verified_scope, "local_adapter_transport_tool_accounting")
  assert.equal(receipt.logical_call_count, 2)
  assert.equal(receipt.attempt_count, 2)
  assert.equal(receipt.retry_count, 0)
  assert.deepEqual(receipt.external_provider_evidence, {
    local_request_header_count: 2,
    local_attempt_count: 2,
    session_title_source: "fallback",
  })
  assert.deepEqual(receipt.observer_checkpoint_contract, {
    evidence_scope: "preflight_probe",
    checkpoint_interval_bytes: 4,
    checkpoint_policy_id: "dsh-text-chunk-at-least-4-utf8-bytes-v1",
    frozen_collection_checkpoint_interval_bytes: 128,
    frozen_collection_checkpoint_policy_id: "dsh-text-chunk-at-least-128-utf8-bytes-v1",
    frozen_collection_policy_verified: false,
  })
  assert.deepEqual(receipt.call_usage, [
    { input_tokens: 10, output_tokens: 5 },
    { input_tokens: 20, output_tokens: 3 },
  ])
  assert.equal(receipt.terminal_instruction_compliant, false)
  assert.equal(
    receipt.observed_final_response_contract,
    "visible_prefix_then_unique_marker_suffix",
  )
  assert.equal(JSON.stringify(receipt).includes(fixtureEvidence.sessionId), false)
})

test("accepts a tool-only first Call with no visible checkpoint", () => {
  const evidence = makeFixtureEvidence({ toolOnlyFirstCall: true })
  const receipt = verifyQwenPreflight(evidence)
  assert.equal(receipt.status, "passed")
  assert.deepEqual(receipt.observer_checkpoint_count, [0, 1])
})

test("rejects a header-only frozen collection checkpoint claim", () => {
  const observerText = rewriteJsonl(fixtureEvidence.observerText, (rows) => {
    rows[0].checkpoint_interval_bytes = 128
    rows[0].checkpoint_policy_id = "dsh-text-chunk-at-least-128-utf8-bytes-v1"
  })
  assert.throws(
    () => verifyQwenPreflight({
      sessionText: fixtureEvidence.sessionText,
      observerText,
      checkpointIntervalBytes: 128,
    }),
    /checkpoint cadence is below the declared interval/,
  )
})

test("rejects session and user provenance drift", () => {
  for (const mutate of [
    (rows) => { rows[0].version = 1 },
    (rows) => {
      rows.find((row) => row.data?.source?.kind === "user").data.source = {
        kind: "plugin",
        plugin: "router-bootstrap",
      }
    },
    (rows) => {
      rows.find((row) => row.type === "session/title").data.source.kind = "model"
    },
  ]) {
    const sessionText = rewriteJsonl(fixtureEvidence.sessionText, mutate)
    assert.throws(
      () => verifyQwenPreflight({ sessionText, observerText: fixtureEvidence.observerText }),
      /session schema version drifted|direct user task|fallback path/,
    )
  }
})

test("rejects stale guidance for tools removed from the Request", () => {
  const sessionText = rewriteJsonl(fixtureEvidence.sessionText, (rows) => {
    rows.find((row) => row.type === "request/header").data.header.system +=
      "\n\nUse the web_search tool to discover current information."
  })
  assert.throws(
    () => verifyQwenPreflight({ sessionText, observerText: fixtureEvidence.observerText }),
    /removed tool guidance leaked/,
  )
})

test("rejects observer schema and plugin compatibility drift", () => {
  for (const mutate of [
    (rows) => { rows[1].schema_version = 1 },
    (rows) => { rows[0].plugin_version = "0.5.0" },
    (rows) => { rows[0].dsh_llm_compatibility = ">=0.2.0" },
  ]) {
    const observerText = rewriteJsonl(fixtureEvidence.observerText, mutate)
    assert.throws(
      () => verifyQwenPreflight({ sessionText: fixtureEvidence.sessionText, observerText }),
      /observer schema version drifted|observer plugin compatibility drifted/,
    )
  }
})

test("rejects observer request identity drift", () => {
  const observerText = rewriteJsonl(fixtureEvidence.observerText, (rows) => {
    rows.filter((row) => row.record_type === "attempt/start")[1].request_id =
      "detached-request"
  })
  assert.throws(
    () => verifyQwenPreflight({ sessionText: fixtureEvidence.sessionText, observerText }),
    /request id drifted/,
  )
})

test("rejects orphan observer checkpoints", () => {
  const observerText = rewriteJsonl(fixtureEvidence.observerText, (rows) => {
    rows.push({
      schema_version: 2,
      record_type: "attempt/checkpoint",
      time: 60,
      session_id: fixtureEvidence.sessionId,
      attempt_id: "orphan-attempt",
    })
  })
  assert.throws(
    () => verifyQwenPreflight({ sessionText: fixtureEvidence.sessionText, observerText }),
    /orphan checkpoint/,
  )
})

test("rejects any llm retry-family event", () => {
  const sessionText = rewriteJsonl(fixtureEvidence.sessionText, (rows) => {
    rows.splice(-1, 0, {
      type: "llm/retry/scheduled",
      seq: -1,
      time: 61,
      data: { turn: 1, step: 2 },
    })
    rows.slice(1).forEach((row, index) => { row.seq = index })
  })
  assert.throws(
    () => verifyQwenPreflight({ sessionText, observerText: fixtureEvidence.observerText }),
    /retry event is forbidden/,
  )
})

test("accepts supplied installed-DSH local-Qwen evidence", skipWithoutEvidence, () => {
  const receipt = verifyQwenPreflight(evidence)
  assert.deepEqual(receipt, {
    status: "passed",
    verified_scope: "local_adapter_transport_tool_accounting",
    external_provider_request_count: 0,
    external_provider_evidence: {
      local_request_header_count: 2,
      local_attempt_count: 2,
      session_title_source: "fallback",
    },
    observer_checkpoint_contract: {
      evidence_scope: "preflight_probe",
      checkpoint_interval_bytes: 4,
      checkpoint_policy_id: "dsh-text-chunk-at-least-4-utf8-bytes-v1",
      frozen_collection_checkpoint_interval_bytes: 128,
      frozen_collection_checkpoint_policy_id: "dsh-text-chunk-at-least-128-utf8-bytes-v1",
      frozen_collection_policy_verified: false,
    },
    logical_call_count: 2,
    attempt_count: 2,
    retry_count: 0,
    agent_preset: "router-standard",
    workspace_identity_verified: true,
    permission_preset: "unrestricted",
    sandbox_mode: "danger-full-access",
    approval_policy: "never",
    first_request_tool_count: 4,
    promoted_request_tool_count: 12,
    tool_call_count: 1,
    tool_name: "pwsh",
    final_marker: "QWEN_DSH_PREFLIGHT_DONE",
    final_response_contract: "ends_with_unique_marker",
    terminal_instruction_compliant: true,
    observed_final_response_contract: "unique_marker_only",
    effective_condition: {
      provider: "local-qwen",
      model: "Qwen/Qwen2.5-7B-Instruct",
      reasoning_effort: "off",
      temperature: 0,
      max_tokens: 192,
      stop: [],
    },
    model_revision: "a09a35458c702b33eeacc393d103063234e8bc28",
    quantization: {
      precision: "bitsandbytes-nf4-4bit",
      quant_type: "nf4",
      double_quant: true,
      compute_dtype: "torch.bfloat16",
      linear4bit_modules: 196,
      lm_head_dtype: "torch.bfloat16",
    },
    call_usage: [
      { input_tokens: 2063, output_tokens: 45 },
      { input_tokens: 4188, output_tokens: 9 },
    ],
    observer_checkpoint_count: [0, 1],
    acknowledged_delta_count: [15, 1],
    session_packed_row_count: 0,
  })
  const serialized = JSON.stringify(receipt)
  assert.equal(serialized.includes(evidence.sessionId), false)
  assert.equal(serialized.includes(EVIDENCE_ROOT), false)
  assert.deepEqual(verifyQwenPreflight(evidence), receipt)
})

test("rejects an observer JSONL file without its final LF", () => {
  assert.throws(
    () => verifyQwenPreflight({
      sessionText: fixtureEvidence.sessionText,
      observerText: fixtureEvidence.observerText.slice(0, -1),
    }),
    /observer JSONL must end with LF/,
  )
})

test("rejects router promotion surface drift", () => {
  const sessionText = rewriteJsonl(fixtureEvidence.sessionText, (rows) => {
    const headers = rows.filter((row) => row.type === "request/header")
    headers[1].data.header.tools.pop()
  })
  assert.throws(
    () => verifyQwenPreflight({ sessionText, observerText: fixtureEvidence.observerText }),
    /no-delegation promoted surface/,
  )
})

test("rejects retry semantics inside either logical Call", () => {
  const observerText = rewriteJsonl(fixtureEvidence.observerText, (rows) => {
    const secondStart = rows.filter((row) => row.record_type === "attempt/start")[1]
    secondStart.attempt_ordinal = 2
    secondStart.retry_of_attempt_id = "fixture-retry-parent"
  })
  assert.throws(
    () => verifyQwenPreflight({ sessionText: fixtureEvidence.sessionText, observerText }),
    /retry is forbidden/,
  )
})

test("rejects a changed pwsh command", () => {
  const sessionText = rewriteJsonl(fixtureEvidence.sessionText, (rows) => {
    const toolCall = rows.find((row) => row.type === "tool/call")
    const argumentsValue = JSON.parse(toolCall.data.arguments)
    argumentsValue.command = "Write-Output WRONG_MARKER"
    toolCall.data.arguments = JSON.stringify(argumentsValue)
    const assistantMessage = rows.find(
      (row) => row.type === "assistant/message" && row.data.step === 1,
    )
    assistantMessage.data.message.content.find(
      (block) => block.type === "tool-call",
    ).arguments = toolCall.data.arguments
  })
  assert.throws(
    () => verifyQwenPreflight({ sessionText, observerText: fixtureEvidence.observerText }),
    /pwsh arguments drifted/,
  )
})

test("rejects pwsh result error and source linkage drift", () => {
  for (const mutate of [
    (result) => { result.data.error = { message: "fixture executor error" } },
    (result) => { result.sourceEventSeqs = [0] },
  ]) {
    const sessionText = rewriteJsonl(fixtureEvidence.sessionText, (rows) => {
      mutate(rows.find((row) => row.type === "tool/result"))
    })
    assert.throws(
      () => verifyQwenPreflight({ sessionText, observerText: fixtureEvidence.observerText }),
      /executor error|source event linkage drifted/,
    )
  }
})

test("rejects native tool-call stream drift", () => {
  const sessionText = rewriteJsonl(fixtureEvidence.sessionText, (rows) => {
    const delta = rows.find(
      (row) => row.type === "assistant/chunk" && row.data.chunk.type === "tool-call-delta",
    )
    delta.data.chunk.argumentsDelta = "{}"
  })
  assert.throws(
    () => verifyQwenPreflight({ sessionText, observerText: fixtureEvidence.observerText }),
    /native streamed tool-call differs/,
  )
})

test("rejects three-way usage drift", () => {
  const observerText = rewriteJsonl(fixtureEvidence.observerText, (rows) => {
    const secondEnd = rows.filter((row) => row.record_type === "attempt/end")[1]
    secondEnd.usage.outputTokens += 1
  })
  assert.throws(
    () => verifyQwenPreflight({ sessionText: fixtureEvidence.sessionText, observerText }),
    /observer usage differs from session usage/,
  )
})

test("rejects cache and reasoning usage drift in every accounting channel", () => {
  for (const field of ["cacheReadTokens", "cacheWriteTokens", "reasoningTokens"]) {
    const observerText = rewriteJsonl(fixtureEvidence.observerText, (rows) => {
      rows.find((row) => row.record_type === "attempt/end").usage[field] = 1
    })
    assert.throws(
      () => verifyQwenPreflight({ sessionText: fixtureEvidence.sessionText, observerText }),
      /observer usage.*null cache and reasoning fields/,
    )
  }
})

test("rejects non-null local cache and reasoning usage even when all channels agree", () => {
  for (const field of ["cacheReadTokens", "cacheWriteTokens", "reasoningTokens"]) {
    const sessionText = rewriteJsonl(fixtureEvidence.sessionText, (rows) => {
      rows.find(
        (row) => row.type === "assistant/chunk"
          && row.data.step === 1
          && row.data.chunk.type === "usage",
      ).data.chunk.usage[field] = 1
      rows.find(
        (row) => row.type === "assistant/message" && row.data.step === 1,
      ).data.usage[field] = 1
    })
    const observerText = rewriteJsonl(fixtureEvidence.observerText, (rows) => {
      rows.find((row) => row.record_type === "attempt/end").usage[field] = 1
    })
    assert.throws(
      () => verifyQwenPreflight({ sessionText, observerText }),
      /usage.*null cache and reasoning fields/,
    )
  }
})

test("rejects non-positive or unsafe provider usage", () => {
  const sessionText = rewriteJsonl(fixtureEvidence.sessionText, (rows) => {
    const usage = rows.find(
      (row) => row.type === "assistant/chunk"
        && row.data.step === 1
        && row.data.chunk.type === "usage",
    )
    const message = rows.find(
      (row) => row.type === "assistant/message" && row.data.step === 1,
    )
    usage.data.chunk.usage.outputTokens = 0
    message.data.usage.outputTokens = 0
  })
  assert.throws(
    () => verifyQwenPreflight({ sessionText, observerText: fixtureEvidence.observerText }),
    /usage must contain positive safe input\/output integers/,
  )
})

test("rejects tokenizer replay count drift", () => {
  const sessionText = rewriteJsonl(fixtureEvidence.sessionText, (rows) => {
    const finish = rows.find(
      (row) => row.type === "assistant/chunk"
        && row.data.step === 2
        && row.data.chunk.type === "finish",
    )
    const message = rows.find(
      (row) => row.type === "assistant/message" && row.data.step === 2,
    )
    finish.data.chunk.replayState.response.tokenizerOutputTokens += 1
    message.data.message.source.replayState.response.tokenizerOutputTokens += 1
  })
  assert.throws(
    () => verifyQwenPreflight({ sessionText, observerText: fixtureEvidence.observerText }),
    /tokenizer replay counts differ from usage/,
  )
})

test("rejects generation without acknowledged online deltas", () => {
  const sessionText = rewriteJsonl(fixtureEvidence.sessionText, (rows) => {
    const finish = rows.find(
      (row) => row.type === "assistant/chunk"
        && row.data.step === 2
        && row.data.chunk.type === "finish",
    )
    const message = rows.find(
      (row) => row.type === "assistant/message" && row.data.step === 2,
    )
    finish.data.chunk.replayState.response.acknowledgedDeltaCount = 0
    message.data.message.source.replayState.response.acknowledgedDeltaCount = 0
  })
  assert.throws(
    () => verifyQwenPreflight({ sessionText, observerText: fixtureEvidence.observerText }),
    /lacks acknowledged online generation deltas/,
  )
})

test("rejects a weakened NF4 replay attestation", () => {
  const sessionText = rewriteJsonl(fixtureEvidence.sessionText, (rows) => {
    const finish = rows.find(
      (row) => row.type === "assistant/chunk"
        && row.data.step === 1
        && row.data.chunk.type === "finish",
    )
    const message = rows.find(
      (row) => row.type === "assistant/message" && row.data.step === 1,
    )
    finish.data.chunk.replayState.response.quantType = "fp16"
    message.data.message.source.replayState.response.quantType = "fp16"
  })
  assert.throws(
    () => verifyQwenPreflight({ sessionText, observerText: fixtureEvidence.observerText }),
    /NF4 attestation drifted/,
  )
})

test("rejects incompatible generation configuration", () => {
  const mutations = [
    (response) => { response.repetitionPenalty = 1 },
    (response) => { response.eosTokenIds = [151645] },
  ]
  for (const mutate of mutations) {
    const sessionText = rewriteJsonl(fixtureEvidence.sessionText, (rows) => {
      const finish = rows.find(
        (row) => row.type === "assistant/chunk"
          && row.data.step === 1
          && row.data.chunk.type === "finish",
      )
      const message = rows.find(
        (row) => row.type === "assistant/message" && row.data.step === 1,
      )
      mutate(finish.data.chunk.replayState.response)
      mutate(message.data.message.source.replayState.response)
    })
    assert.throws(
      () => verifyQwenPreflight({ sessionText, observerText: fixtureEvidence.observerText }),
      /NF4 attestation drifted/,
    )
  }
})

test("rejects observer checkpoint replay drift", () => {
  const observerText = rewriteJsonl(fixtureEvidence.observerText, (rows) => {
    const firstEnd = rows.find(
      (row) => row.record_type === "attempt/end" && row.step === 1,
    )
    firstEnd.visible_text_tail = "fixture-drift"
    firstEnd.observer_visible_output_tokens += Buffer.byteLength("fixture-drift", "utf8")
  })
  assert.throws(
    () => verifyQwenPreflight({ sessionText: fixtureEvidence.sessionText, observerText }),
    /checkpoint replay differs from visible assistant text/,
  )
})

test("rejects a changed pwsh success marker", () => {
  const sessionText = rewriteJsonl(fixtureEvidence.sessionText, (rows) => {
    const result = rows.find((row) => row.type === "tool/result")
    result.data.message.content[0].content[0].text = "WRONG_MARKER\r\n"
  })
  assert.throws(
    () => verifyQwenPreflight({ sessionText, observerText: fixtureEvidence.observerText }),
    /pwsh result must contain only the successful marker/,
  )
})

test("rejects a duplicated terminal marker", () => {
  const sessionText = rewriteJsonl(fixtureEvidence.sessionText, (rows) => {
    const duplicate = "QWEN_DSH_PREFLIGHT_DONE"
    const lastDelta = rows.filter(
      (row) => row.type === "assistant/chunk"
        && row.data.step === 2
        && row.data.chunk.type === "text-delta",
    ).at(-1)
    lastDelta.data.chunk.text += duplicate
    const blockEnd = rows.find(
      (row) => row.type === "assistant/chunk"
        && row.data.step === 2
        && row.data.chunk.type === "block-end",
    )
    blockEnd.data.chunk.block.text += duplicate
    const message = rows.find(
      (row) => row.type === "assistant/message" && row.data.step === 2,
    )
    message.data.message.content[0].text += duplicate
  })
  assert.throws(
    () => verifyQwenPreflight({ sessionText, observerText: fixtureEvidence.observerText }),
    /completion marker must occur exactly once/,
  )
})
