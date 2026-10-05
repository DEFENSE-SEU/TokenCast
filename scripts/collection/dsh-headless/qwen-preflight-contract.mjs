import { isDeepStrictEqual } from "node:util"

import {
  ALLOWED_TOOL_NAMES,
  TOOL_PROMPT_SECTION,
} from "./tool-policy.mjs"

export const QWEN_PREFLIGHT_TASK =
  "Call the pwsh tool exactly once with command Write-Output QWEN_DSH_PREFLIGHT_TOOL_OK and description Print the Qwen DSH preflight marker. After the tool succeeds, reply with exactly QWEN_DSH_PREFLIGHT_DONE. Do not call any other tool."
export const QWEN_PREFLIGHT_CHECKPOINT_INTERVAL_BYTES = 4
export const QWEN_PREFLIGHT_CHECKPOINT_POLICY_ID =
  "dsh-text-chunk-at-least-4-utf8-bytes-v1"

const PROVIDER = "local-qwen"
const MODEL = "Qwen/Qwen2.5-7B-Instruct"
const REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
const TOOL_MARKER = "QWEN_DSH_PREFLIGHT_TOOL_OK"
const FINAL_MARKER = "QWEN_DSH_PREFLIGHT_DONE"
const TOOL_ARGUMENTS = {
  command: `Write-Output ${TOOL_MARKER}`,
  description: "Print the Qwen DSH preflight marker",
}
const EFFECTIVE_CONDITION = {
  provider: PROVIDER,
  model: MODEL,
  reasoningEffort: "off",
  temperature: 0,
  maxTokens: 192,
  stop: [],
}
const INITIAL_TOOLS = ["edit", "pwsh", "read", "write"]
const PROMOTED_TOOLS = [...ALLOWED_TOOL_NAMES]
const REMOVED_TOOL_GUIDANCE = [
  "Use the web_search tool",
  "Use the workflow tool",
  "Use the ralph tool",
  "Use subagent in the background",
  "Use subagent_fork in the background",
  "Use goal tools",
]
const PROBE_CAPTURE_REVISION_PREFIX = "local-qwen-generation-probe-v2:preflight"
const PROBE_CAPTURE_REVISION_POLICY = "prefix-plus-session-id"
const PROBE_SESSION_KEYS = [
  "schema_version",
  "record_type",
  "contract_id",
  "capture_revision",
  "source_id",
  "session_id",
  "model_id",
  "model_revision",
  "projection",
  "causal_alignment",
]
const PROBE_CHECKPOINT_KEYS = [
  "schema_version",
  "record_type",
  "session_id",
  "attempt_id",
  "checkpoint_ordinal",
  "observer_committed_tokens",
  "source_output_tokens",
  "hidden_state_projection",
]
const PROBE_PROJECTION = {
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
}

function invariant(condition, message) {
  if (!condition) throw new Error(`Qwen preflight contract: ${message}`)
}

function equal(actual, expected, message) {
  invariant(isDeepStrictEqual(actual, expected), message)
}

function exactKeys(value, expected, message) {
  invariant(value !== null && typeof value === "object" && !Array.isArray(value), message)
  equal(Object.keys(value).sort(), [...expected].sort(), message)
}

function parseJsonl(text, label) {
  invariant(typeof text === "string", `${label} must be UTF-8 text`)
  invariant(text.length > 0, `${label} is empty`)
  invariant(text.endsWith("\n"), `${label} must end with LF`)
  const lines = text.slice(0, -1).split("\n")
  invariant(lines.every((line) => line.length > 0), `${label} contains a blank line`)
  return lines.map((line, index) => {
    try {
      return JSON.parse(line)
    } catch (error) {
      throw new Error(`${label} line ${index + 1} is invalid JSON: ${error.message}`)
    }
  })
}

function probeJoinKey(row) {
  return [
    row.session_id,
    row.attempt_id,
    row.checkpoint_ordinal,
    row.observer_committed_tokens,
  ].join("\u0000")
}

export function verifyGenerationProbePreflight({ observerText, probeText }) {
  const observerRows = parseJsonl(observerText, "observer JSONL")
  const probeRows = parseJsonl(probeText, "Generation Probe JSONL")
  const observerSession = observerRows[0]
  invariant(observerSession?.record_type === "observer/session", "observer session header is missing")
  const header = probeRows[0]
  exactKeys(header, PROBE_SESSION_KEYS, "Generation Probe session fields drifted")
  invariant(
    header.schema_version === 2
      && header.record_type === "generation_probe/session"
      && header.contract_id === "dsh-generation-probe-v2"
      && header.capture_revision
        === `${PROBE_CAPTURE_REVISION_PREFIX}:${observerSession.session_id}`
      && header.source_id === "dsh_session_probe_v2"
      && header.session_id === observerSession.session_id
      && header.model_id === MODEL
      && header.model_revision === REVISION
      && header.causal_alignment === "post_source_output_token_prefix_before_next_sample",
    "Generation Probe session contract drifted",
  )
  exactKeys(
    header.projection,
    Object.keys(PROBE_PROJECTION),
    "Generation Probe projection fields drifted",
  )
  equal(header.projection, PROBE_PROJECTION, "Generation Probe projection policy drifted")

  const checkpoints = probeRows.slice(1)
  const terminalOutputByAttempt = new Map(
    observerRows
      .filter((row) => row.record_type === "attempt/end")
      .map((row) => [row.attempt_id, usageVector(row.usage).outputTokens]),
  )
  const checkpointsByAttempt = new Map()
  const previousCheckpointByAttempt = new Map()
  for (const checkpoint of checkpoints) {
    exactKeys(
      checkpoint,
      PROBE_CHECKPOINT_KEYS,
      "Generation Probe checkpoint fields drifted",
    )
    invariant(
      checkpoint.schema_version === 2
        && checkpoint.record_type === "generation_probe/checkpoint"
        && checkpoint.session_id === header.session_id
        && typeof checkpoint.attempt_id === "string"
        && checkpoint.attempt_id.length > 0
        && Number.isSafeInteger(checkpoint.checkpoint_ordinal)
        && checkpoint.checkpoint_ordinal > 0
        && Number.isSafeInteger(checkpoint.observer_committed_tokens)
        && checkpoint.observer_committed_tokens > 0
        && Number.isSafeInteger(checkpoint.source_output_tokens)
        && checkpoint.source_output_tokens > 0,
      "Generation Probe checkpoint identity drifted",
    )
    const attemptCheckpoints = checkpointsByAttempt.get(checkpoint.attempt_id) ?? []
    attemptCheckpoints.push(checkpoint)
    checkpointsByAttempt.set(checkpoint.attempt_id, attemptCheckpoints)
    const previousCheckpoint = previousCheckpointByAttempt.get(checkpoint.attempt_id)
    invariant(
      checkpoint.checkpoint_ordinal
          === (previousCheckpoint === undefined ? 1 : previousCheckpoint.checkpoint_ordinal + 1)
        && (
          previousCheckpoint === undefined
          || checkpoint.observer_committed_tokens > previousCheckpoint.observer_committed_tokens
        ),
      "Generation Probe checkpoint order drifted within an Attempt",
    )
    previousCheckpointByAttempt.set(checkpoint.attempt_id, checkpoint)
    const terminalOutput = terminalOutputByAttempt.get(checkpoint.attempt_id)
    invariant(
      Number.isSafeInteger(terminalOutput)
        && checkpoint.source_output_tokens <= terminalOutput,
      "Generation Probe source output position exceeds terminal usage",
    )
    invariant(
      Array.isArray(checkpoint.hidden_state_projection)
        && checkpoint.hidden_state_projection.length === PROBE_PROJECTION.dimension
        && checkpoint.hidden_state_projection.every(Number.isFinite),
      "Generation Probe checkpoint projection drifted",
    )
  }
  for (const attemptCheckpoints of checkpointsByAttempt.values()) {
    const ordered = [...attemptCheckpoints].sort(
      (left, right) => left.checkpoint_ordinal - right.checkpoint_ordinal,
    )
    for (let index = 1; index < ordered.length; index += 1) {
      invariant(
        ordered[index].source_output_tokens >= ordered[index - 1].source_output_tokens,
        "Generation Probe source output position decreased within an Attempt",
      )
    }
  }

  const observerKeys = observerRows
    .filter((row) => row.record_type === "attempt/checkpoint")
    .map(probeJoinKey)
  const probeKeys = checkpoints.map(probeJoinKey)
  equal(
    [...probeKeys].sort(),
    [...observerKeys].sort(),
    "Generation Probe checkpoints differ from observer checkpoints",
  )
  equal(
    probeKeys,
    observerKeys,
    "Generation Probe checkpoint order differs from observer checkpoints",
  )
  return {
    captureRevisionPrefix: PROBE_CAPTURE_REVISION_PREFIX,
    captureRevisionPolicy: PROBE_CAPTURE_REVISION_POLICY,
    checkpointCount: checkpoints.length,
    projectionPolicyId: header.projection.policy_id,
    projectionDimension: header.projection.dimension,
    causalAlignment: header.causal_alignment,
    sourceOutputTokens: checkpoints.map((checkpoint) => checkpoint.source_output_tokens),
  }
}

function contentText(content) {
  return content
    .filter((block) => block?.type === "text")
    .map((block) => block.text)
    .join("")
}

function countOccurrences(text, needle) {
  let count = 0
  let offset = 0
  while ((offset = text.indexOf(needle, offset)) !== -1) {
    count += 1
    offset += needle.length
  }
  return count
}

function stepRows(rows, type, step) {
  return rows.filter((row) => row.type === type && row.data?.turn === 1 && row.data?.step === step)
}

function usageVector(usage) {
  return {
    inputTokens: usage?.inputTokens,
    outputTokens: usage?.outputTokens,
    cacheReadTokens: usage?.cacheReadTokens ?? null,
    cacheWriteTokens: usage?.cacheWriteTokens ?? null,
    reasoningTokens: usage?.reasoningTokens ?? null,
  }
}

function verifyLocalUsage(usage, label) {
  invariant(
    Number.isSafeInteger(usage.inputTokens)
      && usage.inputTokens > 0
      && Number.isSafeInteger(usage.outputTokens)
      && usage.outputTokens > 0,
    `${label} must contain positive safe input/output integers`,
  )
  invariant(
    usage.cacheReadTokens === null
      && usage.cacheWriteTokens === null
      && usage.reasoningTokens === null,
    `${label} must report null cache and reasoning fields for the local uncached non-reasoning route`,
  )
}

function verifySession(sessionText, decodeStorageRecord, requirePackedSession, expectedWorkspace) {
  const rows = parseJsonl(sessionText, "session JSONL")
  const session = rows[0]
  invariant(session?.type === "session" && session.seq === undefined, "first session row must be the unsequenced session header")
  invariant(session.version === 0, "session schema version drifted")
  invariant(typeof session.id === "string" && session.id.length > 0, "session header id is missing")
  invariant(session.agentPreset === "router-standard", "router-standard preset is missing")
  invariant(typeof session.cwd === "string" && session.cwd.length > 0, "session workspace is missing")
  invariant(
    expectedWorkspace === undefined || session.cwd === expectedWorkspace,
    "session workspace differs from the preflight workspace",
  )

  const storageRows = rows.slice(1)
  const packedTypes = new Set(["text-chunks", "reasoning-chunks", "tool-call-chunks"])
  const packedRowCount = storageRows.filter((row) => packedTypes.has(row?.type)).length
  invariant(!requirePackedSession || packedRowCount > 0, "session contains no packed chunk row")
  invariant(
    packedRowCount === 0 || typeof decodeStorageRecord === "function",
    "packed session requires the DSH storage decoder",
  )
  let events
  try {
    events = storageRows.flatMap((row) => (
      typeof decodeStorageRecord === "function" ? decodeStorageRecord(row) : [row]
    ))
  } catch (error) {
    throw new Error(`Qwen preflight contract: session storage row is invalid: ${error.message}`)
  }
  events.forEach((row, index) => {
    invariant(row.seq === index, `session seq must be continuous at event ${index}`)
  })
  const permissionEvents = events.filter((event) => (
    event.type === "permission/preset"
    || event.type === "sandbox/mode"
    || event.type === "approval/policy"
  ))
  equal(permissionEvents.map((event) => [event.type, event.data]), [
    ["permission/preset", { preset: "unrestricted" }],
    ["sandbox/mode", { mode: "danger-full-access" }],
    ["approval/policy", { policy: "never" }],
  ], "session permission facts differ from the frozen actor policy")

  const taskMessages = events.filter(
    (row) => row.type === "user/message" && row.data?.source?.kind === "user",
  )
  invariant(taskMessages.length === 1, "expected one direct user task message")
  equal(taskMessages[0].data.content, [{ type: "text", text: QWEN_PREFLIGHT_TASK }], "preflight task text drifted")
  invariant(typeof taskMessages[0].data.id === "string", "direct user task id is missing")
  const routerGuides = events.filter(
    (row) => row.type === "user/message" && row.data?.source?.plugin === "router-bootstrap",
  )
  invariant(routerGuides.length === 0, "direct execution must use the react route without a guide")
  const titles = events.filter((row) => row.type === "session/title")
  invariant(
    titles.length === 1
      && titles[0].data?.source?.kind === "fallback"
      && isDeepStrictEqual(titles[0].data?.messageSeqs, [taskMessages[0].seq]),
    "session title must come from the fallback path",
  )

  const starts = events.filter((row) => row.type === "step/start")
  const ends = events.filter((row) => row.type === "step/end")
  equal(starts.map((row) => [row.data?.turn, row.data?.step]), [[1, 1], [1, 2]], "expected exactly two ordered Call starts")
  equal(ends.map((row) => [row.data?.turn, row.data?.step]), [[1, 1], [1, 2]], "expected exactly two ordered Call ends")
  invariant(events.every((row) => !row.type?.startsWith("llm/retry")), "retry event is forbidden")
  invariant(events.every((row) => !row.type?.includes("error")), "session contains an error event")

  const headers = events.filter((row) => row.type === "request/header")
  invariant(headers.length === 2, "expected two Request headers")
  equal(headers.map((row) => row.data?.reason), ["initial", "change"], "Request reasons must seal initial routing and promotion")
  const expectedConfig = {
    provider: PROVIDER,
    model: MODEL,
    maxTokens: EFFECTIVE_CONDITION.maxTokens,
    temperature: EFFECTIVE_CONDITION.temperature,
    stop: EFFECTIVE_CONDITION.stop,
    reasoningEffort: EFFECTIVE_CONDITION.reasoningEffort,
  }
  for (const header of headers) {
    equal(header.data?.header?.config, expectedConfig, "effective local-Qwen condition drifted")
    const system = header.data?.header?.system
    invariant(
      typeof system === "string" && system.includes(TOOL_PROMPT_SECTION.text),
      "DSH tool-surface prompt policy is missing",
    )
    invariant(
      REMOVED_TOOL_GUIDANCE.every((guidance) => !system.includes(guidance)),
      "removed tool guidance leaked into the system prompt",
    )
  }
  const headerToolNames = headers.map((row) => row.data?.header?.tools?.map((tool) => tool.name))
  equal(headerToolNames[0], INITIAL_TOOLS, "initial router surface must expose the four declared tools")
  equal(headerToolNames[1], PROMOTED_TOOLS, "second Request must expose the declared no-delegation promoted surface")
  equal(
    headers.map((row) => row.data.header.system.includes("Track every background job id you start.")),
    [false, true],
    "grouped job guidance must follow the promoted job tools",
  )

  const contexts = events.filter((row) => row.type === "request/context")
  invariant(contexts.length === 1, "expected one local context-window declaration")
  invariant(
    contexts[0].data?.provider === PROVIDER
      && contexts[0].data?.model === MODEL
      && contexts[0].data?.contextWindow === 32768,
    "local Qwen context declaration drifted",
  )

  const messages = events.filter(
    (row) => row.type === "assistant/message" && row.data?.message?.source?.kind === "model",
  )
  invariant(messages.length === 2, "expected two model assistant messages")
  equal(messages.map((row) => [row.data?.turn, row.data?.step]), [[1, 1], [1, 2]], "assistant messages crossed Call boundaries")
  for (const message of messages) {
    invariant(
      message.data.message.source.provider === PROVIDER
        && message.data.message.source.model === MODEL,
      "assistant message used an unexpected provider route",
    )
  }

  const toolCalls = events.filter((row) => row.type === "tool/call")
  const toolResults = events.filter((row) => row.type === "tool/result")
  const assistantToolBlocks = messages.flatMap((row) =>
    row.data.message.content.filter((block) => block.type === "tool-call"),
  )
  invariant(toolCalls.length === 1 && toolResults.length === 1 && assistantToolBlocks.length === 1, "expected one complete tool-call chain")
  const assistantTool = assistantToolBlocks[0]
  const toolCall = toolCalls[0]
  const toolResult = toolResults[0]
  invariant(assistantTool.name === "pwsh" && toolCall.data?.name === "pwsh", "the sole tool call must use pwsh")
  invariant(
    assistantTool.id === toolCall.data?.callId
      && toolCall.data.callId === toolResult.data?.message?.source?.callId,
    "pwsh callId linkage is broken",
  )
  invariant(assistantTool.arguments === toolCall.data.arguments, "assistant and tool-event arguments differ")
  let parsedArguments
  try {
    parsedArguments = JSON.parse(toolCall.data.arguments)
  } catch (error) {
    throw new Error(`Qwen preflight contract: pwsh arguments are invalid JSON: ${error.message}`)
  }
  equal(parsedArguments, TOOL_ARGUMENTS, "pwsh arguments drifted")
  invariant(toolCall.data.turn === 1 && toolCall.data.step === 1, "pwsh must execute inside the first Call")

  const resultMessage = toolResult.data?.message
  const resultBlocks = resultMessage?.content
  invariant(
    toolResult.data?.turn === 1
      && toolResult.data?.step === 1
      && resultMessage?.role === "user"
      && resultMessage?.source?.kind === "tool"
      && Array.isArray(resultBlocks)
      && resultBlocks.length === 1,
    "pwsh result envelope drifted",
  )
  const result = resultBlocks[0]
  invariant(
    result.type === "tool-result"
      && result.toolCallId === toolCall.data.callId
      && result.isError === false
      && Array.isArray(result.content)
      && result.content.length === 1
      && result.content[0]?.type === "text"
      && /^QWEN_DSH_PREFLIGHT_TOOL_OK\r?\n$/.test(result.content[0].text),
    "pwsh result must contain only the successful marker",
  )
  invariant(toolResult.data.error === undefined, "pwsh result contains an executor error")
  equal(toolResult.sourceEventSeqs, [toolCall.seq], "pwsh result source event linkage drifted")

  const firstContent = messages[0].data.message.content
  invariant(
    firstContent.every((block) => block.type === "text" || block.type === "tool-call")
      && firstContent.at(-1)?.type === "tool-call",
    "first assistant message must end with the pwsh call",
  )
  const assistantToolIndex = firstContent.findIndex((block) => block.type === "tool-call")
  invariant(assistantToolIndex >= 0, "first assistant message has no pwsh call")
  const terminalContent = messages[1].data.message.content
  invariant(terminalContent.length > 0 && terminalContent.every((block) => block.type === "text"), "terminal assistant message may contain visible text only")
  const terminalText = contentText(terminalContent)
  invariant(terminalText.endsWith(FINAL_MARKER), "terminal text must end with the completion marker")
  invariant(countOccurrences(terminalText, FINAL_MARKER) === 1, "terminal completion marker must occur exactly once")

  const calls = [1, 2].map((step, index) => {
    const chunks = stepRows(events, "assistant/chunk", step)
    invariant(chunks.length > 0, `Call ${step} has no streamed chunks`)
    const allowedChunkTypes = new Set([
      "block-start",
      "text-delta",
      "tool-call-delta",
      "block-end",
      "usage",
      "finish",
    ])
    invariant(
      chunks.every((row) => allowedChunkTypes.has(row.data.chunk?.type)),
      `Call ${step} contains an unexpected streamed chunk`,
    )
    const usageChunks = chunks.filter((row) => row.data.chunk?.type === "usage")
    const finishes = chunks.filter((row) => row.data.chunk?.type === "finish")
    invariant(usageChunks.length === 1 && finishes.length === 1, `Call ${step} must have one usage and one finish chunk`)
    invariant(chunks.at(-2) === usageChunks[0] && chunks.at(-1) === finishes[0], `Call ${step} must close with usage then finish`)

    const textDeltas = chunks.filter((row) => row.data.chunk?.type === "text-delta")
    const streamedText = textDeltas.map((row) => row.data.chunk.text).join("")
    const messageText = contentText(messages[index].data.message.content)
    invariant(streamedText === messageText, `Call ${step} streamed text does not replay to its assistant message`)
    const textEnds = chunks.filter(
      (row) => row.data.chunk?.type === "block-end" && row.data.chunk?.block?.type === "text",
    )
    invariant(
      messageText === ""
        ? textDeltas.length === 0 && textEnds.length === 0
        : textDeltas.length > 0
          && textEnds.length === 1
          && textEnds[0].data.chunk.block.text === messageText,
      `Call ${step} text block replay drifted`,
    )
    equal(messages[index].sourceEventSeqs, chunks.map((row) => row.seq), `Call ${step} source event sequence drifted`)

    const usage = usageVector(usageChunks[0].data.chunk.usage)
    verifyLocalUsage(usage, `Call ${step} usage`)
    equal(usageVector(messages[index].data.usage), usage, `Call ${step} message usage drifted`)
    const finish = finishes[0].data.chunk
    const replay = finish.replayState?.response
    equal(messages[index].data.message.source.replayState?.response, replay, `Call ${step} replay state differs between finish and message`)
    invariant(finish.reason?.kind === (step === 1 ? "tool-calls" : "stop"), `Call ${step} finish reason drifted`)
    invariant(replay?.revision === REVISION, `Call ${step} model revision drifted`)
    invariant(
      replay.precision === "bitsandbytes-nf4-4bit"
        && replay.device === "cuda:0"
        && replay.modelClass === "Qwen2Model"
        && replay.isQuantized === true
        && replay.quantType === "nf4"
        && replay.doubleQuant === true
        && replay.computeDtype === "torch.bfloat16"
        && replay.tieWordEmbeddings === false
        && replay.linear4bitModules === 196
        && replay.lmHeadDtype === "torch.bfloat16"
        && replay.repetitionPenalty === 1.05
        && isDeepStrictEqual(replay.eosTokenIds, [151645, 151643]),
      `Call ${step} NF4 attestation drifted`,
    )
    invariant(
      replay.tokenizerInputTokens === usage.inputTokens
        && replay.tokenizerOutputTokens === usage.outputTokens,
      `Call ${step} tokenizer replay counts differ from usage`,
    )
    invariant(
      Number.isInteger(replay.acknowledgedDeltaCount)
        && replay.acknowledgedDeltaCount >= textDeltas.length
        && replay.acknowledgedDeltaCount > 0,
      `Call ${step} lacks acknowledged online generation deltas`,
    )
    const toolChunkStarts = chunks.filter(
      (row) => row.data.chunk?.type === "block-start" && row.data.chunk?.blockType === "tool-call",
    )
    const toolChunkDeltas = chunks.filter((row) => row.data.chunk?.type === "tool-call-delta")
    const toolChunkEnds = chunks.filter(
      (row) => row.data.chunk?.type === "block-end" && row.data.chunk?.block?.type === "tool-call",
    )
    if (step === 1) {
      invariant(
        toolChunkStarts.length === 1
          && toolChunkDeltas.length > 0
          && toolChunkEnds.length === 1,
        "first Call must contain one native streamed tool-call block",
      )
      const streamedTool = toolChunkEnds[0].data.chunk.block
      invariant(
        toolChunkStarts[0].data.chunk.index === assistantToolIndex
          && toolChunkEnds[0].data.chunk.index === assistantToolIndex
          && toolChunkDeltas.every((row) => row.data.chunk.index === assistantToolIndex)
          && streamedTool.id === assistantTool.id
          && streamedTool.name === assistantTool.name
          && streamedTool.arguments === assistantTool.arguments
          && toolChunkDeltas.every(
            (row) => row.data.chunk.id === assistantTool.id && row.data.chunk.name === assistantTool.name,
          )
          && toolChunkDeltas.map((row) => row.data.chunk.argumentsDelta).join("") === assistantTool.arguments,
        "native streamed tool-call differs from the assistant and executor events",
      )
      invariant(
        chunks.indexOf(toolChunkStarts[0]) < chunks.indexOf(toolChunkDeltas[0])
          && chunks.indexOf(toolChunkDeltas.at(-1)) < chunks.indexOf(toolChunkEnds[0])
          && chunks.indexOf(toolChunkEnds[0]) < chunks.indexOf(usageChunks[0]),
        "native streamed tool-call chunks are out of order",
      )
    } else {
      invariant(
        toolChunkStarts.length === 0 && toolChunkDeltas.length === 0 && toolChunkEnds.length === 0,
        "terminal Call must contain zero tool-call chunks",
      )
    }
    return {
      step,
      message: messages[index],
      finish,
      usage,
      replay,
      visibleText: messageText,
      textDeltaCount: textDeltas.length,
    }
  })

  const turnEnds = events.filter((row) => row.type === "turn/end")
  invariant(
    turnEnds.length === 1
      && turnEnds[0] === events.at(-1)
      && turnEnds[0].data?.turn === 1
      && turnEnds[0].data?.reason?.kind === "completed",
    "session must end with one completed turn",
  )

  return {
    rows,
    session,
    calls,
    headers,
    packedRowCount,
    workspaceIdentityVerified: expectedWorkspace !== undefined,
  }
}

function verifyObserver(
  observerText,
  sessionEvidence,
  checkpointIntervalBytes = QWEN_PREFLIGHT_CHECKPOINT_INTERVAL_BYTES,
) {
  const rows = parseJsonl(observerText, "observer JSONL")
  invariant(rows.every((row) => row.schema_version === 2), "observer schema version drifted")
  const observerSessionRows = rows.filter((row) => row.record_type === "observer/session")
  invariant(observerSessionRows.length === 1 && observerSessionRows[0] === rows[0], "observer must begin with one session record")
  const observerSession = observerSessionRows[0]
  invariant(observerSession.session_id === sessionEvidence.session.id, "observer and DSH session ids differ")
  invariant(
    observerSession.plugin_version === "0.4.0"
      && observerSession.dsh_llm_compatibility === ">=0.1.1-rc.2 <0.2.0"
      && observerSession.observer_durability_contract_id === "observer-jsonl-fsync-on-session-flush-v1",
    "observer plugin compatibility drifted",
  )
  invariant(
    observerSession.checkpoint_interval_bytes === checkpointIntervalBytes
      && observerSession.checkpoint_policy_id === `dsh-text-chunk-at-least-${checkpointIntervalBytes}-utf8-bytes-v1`
      && observerSession.observer_unit_contract_id === "observer:utf8-bytes-v1"
      && observerSession.observer_tokenizer_id === "utf8-byte-256"
      && observerSession.observer_normalization_id === "utf16-to-well-formed-v1"
      && observerSession.observer_carry_policy_id === "utf16-high-surrogate-carry-v1"
      && observerSession.observer_boundary_policy_id === "dsh-text-delta-chunk-v1",
    "observer checkpoint contract drifted",
  )
  invariant(rows.every((row) => row.session_id === sessionEvidence.session.id), "observer rows crossed a session boundary")

  const starts = rows.filter((row) => row.record_type === "attempt/start")
  const ends = rows.filter((row) => row.record_type === "attempt/end")
  const allCheckpoints = rows.filter((row) => row.record_type === "attempt/checkpoint")
  invariant(starts.length === 2 && ends.length === 2, "expected two Calls and two Attempts")
  equal(starts.map((row) => [row.turn, row.step]), [[1, 1], [1, 2]], "observer Call starts are out of order")
  equal(ends.map((row) => [row.turn, row.step]), [[1, 1], [1, 2]], "observer Attempt ends are out of order")
  invariant(new Set(starts.map((row) => row.logical_call_id)).size === 2, "logical Calls must remain distinct")
  invariant(starts.every((row) => row.attempt_ordinal === 1 && row.retry_of_attempt_id === null), "retry is forbidden in the Qwen preflight")
  const knownAttemptIds = new Set(starts.map((row) => row.attempt_id))
  invariant(
    allCheckpoints.every((row) => knownAttemptIds.has(row.attempt_id)),
    "observer contains an orphan checkpoint",
  )
  invariant(rows.indexOf(ends[0]) < rows.indexOf(starts[1]), "observer Attempts overlap")

  const providerUnit = "provider:dsh-outputTokens-v1:local-qwen:Qwen%2FQwen2.5-7B-Instruct"
  const checkpointCounts = []
  for (let index = 0; index < 2; index += 1) {
    const call = sessionEvidence.calls[index]
    const start = starts[index]
    const end = ends[index]
    invariant(
      start.provider === PROVIDER
        && start.model === MODEL
        && start.reasoning_effort === "off"
        && start.provider_token_unit_contract_id === providerUnit,
      `Attempt ${index + 1} effective route drifted`,
    )
    invariant(
      start.logical_call_id === end.logical_call_id
        && start.attempt_id === end.attempt_id
        && end.attempt_ordinal === 1
        && end.provider_token_unit_contract_id === providerUnit,
      `Attempt ${index + 1} identity linkage drifted`,
    )
    const expectedRequestId = `${sessionEvidence.session.id}:turn-${start.turn}:step-${start.step}:request`
    invariant(start.request_id === expectedRequestId, `Attempt ${index + 1} request id drifted`)
    invariant(
      start.request_shape?.tool_count === (index === 0 ? INITIAL_TOOLS.length : PROMOTED_TOOLS.length)
        && start.request_shape?.message_count === (index === 0 ? 1 : 3)
        && Number.isInteger(start.request_shape?.system_utf8_bytes)
        && start.request_shape.system_utf8_bytes > 0,
      `Attempt ${index + 1} Request shape drifted`,
    )
    invariant(
      end.finish_reason === (index === 0 ? "tool-calls" : "stop")
        && end.failure_code === null
        && end.observer_finalized === true,
      `Attempt ${index + 1} terminal state drifted`,
    )
    const observerUsage = usageVector(end.usage)
    verifyLocalUsage(observerUsage, `Attempt ${index + 1} observer usage`)
    equal(observerUsage, call.usage, `Attempt ${index + 1} observer usage differs from session usage`)

    const checkpoints = rows.filter(
      (row) => row.record_type === "attempt/checkpoint" && row.attempt_id === start.attempt_id,
    )
    invariant(
      checkpoints.length > 0 || Buffer.byteLength(call.visibleText, "utf8") < checkpointIntervalBytes,
      `Attempt ${index + 1} has no online observer checkpoint`,
    )
    const startIndex = rows.indexOf(start)
    const endIndex = rows.indexOf(end)
    invariant(
      startIndex < endIndex
        && checkpoints.every((checkpoint) => {
          const checkpointIndex = rows.indexOf(checkpoint)
          return checkpointIndex > startIndex && checkpointIndex < endIndex
        }),
      `Attempt ${index + 1} checkpoint records are outside their Attempt boundary`,
    )
    checkpointCounts.push(checkpoints.length)
    let replayed = ""
    let previousTime = start.time
    checkpoints.forEach((checkpoint, checkpointIndex) => {
      invariant(
        checkpoint.turn === start.turn
          && checkpoint.step === start.step
          && checkpoint.logical_call_id === start.logical_call_id
          && checkpoint.attempt_ordinal === 1
          && checkpoint.checkpoint_ordinal === checkpointIndex + 1
          && checkpoint.observer_unit_contract_id === "observer:utf8-bytes-v1",
        `Attempt ${index + 1} checkpoint ${checkpointIndex + 1} linkage drifted`,
      )
      invariant(typeof checkpoint.visible_text_delta === "string" && checkpoint.visible_text_delta.length > 0, `Attempt ${index + 1} contains an empty checkpoint delta`)
      invariant(
        Buffer.byteLength(checkpoint.visible_text_delta, "utf8") >= checkpointIntervalBytes,
        `Attempt ${index + 1} checkpoint cadence is below the declared interval`,
      )
      replayed += checkpoint.visible_text_delta
      invariant(
        checkpoint.observer_committed_tokens === Buffer.byteLength(replayed, "utf8"),
        `Attempt ${index + 1} checkpoint token ledger drifted`,
      )
      invariant(checkpoint.time >= previousTime && checkpoint.time <= end.time, `Attempt ${index + 1} checkpoint time is out of order`)
      previousTime = checkpoint.time
    })
    const tail = end.visible_text_tail
    invariant(typeof tail === "string", `Attempt ${index + 1} visible tail is missing`)
    replayed += tail
    invariant(replayed === call.visibleText, `Attempt ${index + 1} checkpoint replay differs from visible assistant text`)
    invariant(
      end.observer_visible_output_tokens === Buffer.byteLength(replayed, "utf8"),
      `Attempt ${index + 1} observer visible-token ledger drifted`,
    )
    invariant(
      checkpoints.length === 0 || checkpoints[0].time < call.message.time,
      `Attempt ${index + 1} first checkpoint was not recorded online`,
    )
  }
  invariant(starts[0].request_id !== starts[1].request_id, "tool feedback must create the second Request")
  invariant(
    allCheckpoints.length === checkpointCounts.reduce((total, count) => total + count, 0),
    "observer checkpoint inventory drifted",
  )

  const recognized = new Set(["observer/session", "attempt/start", "attempt/checkpoint", "attempt/end"])
  invariant(rows.every((row) => recognized.has(row.record_type)), "observer contains an unexpected record type")
  return {
    rows,
    starts,
    ends,
    checkpointCounts,
    checkpointIntervalBytes: observerSession.checkpoint_interval_bytes,
    checkpointPolicyId: observerSession.checkpoint_policy_id,
  }
}

export function verifyQwenPreflight({
  sessionText,
  observerText,
  checkpointIntervalBytes = QWEN_PREFLIGHT_CHECKPOINT_INTERVAL_BYTES,
  decodeStorageRecord,
  requirePackedSession = false,
  expectedWorkspace,
}) {
  const sessionEvidence = verifySession(
    sessionText,
    decodeStorageRecord,
    requirePackedSession,
    expectedWorkspace,
  )
  const observerEvidence = verifyObserver(observerText, sessionEvidence, checkpointIntervalBytes)
  const terminalText = sessionEvidence.calls[1].visibleText
  const terminalPrefix = terminalText.slice(0, -FINAL_MARKER.length)
  const terminalInstructionCompliant = terminalPrefix.trim().length === 0
  const callUsage = sessionEvidence.calls.map((call) => ({
    input_tokens: call.usage.inputTokens,
    output_tokens: call.usage.outputTokens,
  }))
  return {
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
      checkpoint_interval_bytes: observerEvidence.checkpointIntervalBytes,
      checkpoint_policy_id: observerEvidence.checkpointPolicyId,
      frozen_collection_checkpoint_interval_bytes: 128,
      frozen_collection_checkpoint_policy_id: "dsh-text-chunk-at-least-128-utf8-bytes-v1",
      frozen_collection_policy_verified: observerEvidence.checkpointIntervalBytes === 128,
    },
    logical_call_count: 2,
    attempt_count: 2,
    retry_count: 0,
    agent_preset: "router-standard",
    workspace_identity_verified: sessionEvidence.workspaceIdentityVerified,
    permission_preset: "unrestricted",
    sandbox_mode: "danger-full-access",
    approval_policy: "never",
    first_request_tool_count: INITIAL_TOOLS.length,
    promoted_request_tool_count: PROMOTED_TOOLS.length,
    tool_call_count: 1,
    tool_name: "pwsh",
    final_marker: FINAL_MARKER,
    final_response_contract: "ends_with_unique_marker",
    terminal_instruction_compliant: terminalInstructionCompliant,
    observed_final_response_contract: terminalInstructionCompliant
      ? "unique_marker_only"
      : "visible_prefix_then_unique_marker_suffix",
    effective_condition: {
      provider: EFFECTIVE_CONDITION.provider,
      model: EFFECTIVE_CONDITION.model,
      reasoning_effort: EFFECTIVE_CONDITION.reasoningEffort,
      temperature: EFFECTIVE_CONDITION.temperature,
      max_tokens: EFFECTIVE_CONDITION.maxTokens,
      stop: EFFECTIVE_CONDITION.stop,
    },
    model_revision: REVISION,
    quantization: {
      precision: "bitsandbytes-nf4-4bit",
      quant_type: "nf4",
      double_quant: true,
      compute_dtype: "torch.bfloat16",
      linear4bit_modules: 196,
      lm_head_dtype: "torch.bfloat16",
    },
    call_usage: callUsage,
    observer_checkpoint_count: observerEvidence.checkpointCounts,
    acknowledged_delta_count: sessionEvidence.calls.map(
      (call) => call.replay.acknowledgedDeltaCount,
    ),
    session_packed_row_count: sessionEvidence.packedRowCount,
  }
}
