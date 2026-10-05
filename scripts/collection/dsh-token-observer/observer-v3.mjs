import { Buffer } from "node:buffer"

export const OBSERVER_V3_SCHEMA_VERSION = 3
export const OBSERVER_V3_PLUGIN_VERSION = "0.5.0"
export const DSH_LLM_COMPATIBILITY = ">=0.1.1-rc.2 <0.2.0"
export const OBSERVER_DURABILITY_CONTRACT_ID =
  "observer-jsonl-fsync-on-session-flush-v1"
export const OBSERVER_UNIT_CONTRACT_ID = "observer:utf8-bytes-v1"
export const GENERATION_UNIT_CONTRACT_ID = "observer:generation-utf8-bytes-v2"
export const OBSERVER_TOKENIZER_ID = "utf8-byte-256"
export const OBSERVER_NORMALIZATION_ID = "utf16-to-well-formed-v1"
export const OBSERVER_CARRY_POLICY_ID = "utf16-high-surrogate-carry-v1"
export const OBSERVER_BOUNDARY_POLICY_ID = "dsh-generation-delta-chunk-v2"
export const ATTEMPT_BOUNDARY_CONTRACT_ID =
  "dsh-llm-waterfall-entry-one-observed-attempt-v1"

export const GENERATION_CHANNELS = ["text", "reasoning", "tool_arguments"]

export const NATIVE_RETRY_SCHEDULED = "llm/retry"
export const NATIVE_RETRY_STARTED = "llm/retry-started"
export const NATIVE_REQUEST_HEADER = "request/header"
export const NATIVE_REQUEST_CONTEXT = "request/context"

const CHANNEL_BY_CHUNK_TYPE = {
  "text-delta": "text",
  "reasoning-delta": "reasoning",
  "tool-call-delta": "tool_arguments",
}

export function generationCheckpointPolicyId(checkpointIntervalBytes) {
  return `dsh-generation-chunk-at-least-${checkpointIntervalBytes}-utf8-bytes-v2`
}

function zeroChannels() {
  return { text: 0, reasoning: 0, tool_arguments: 0 }
}

function falseChannels() {
  return { text: false, reasoning: false, tool_arguments: false }
}

function isHighSurrogate(code) {
  return code >= 0xd800 && code <= 0xdbff
}

export function splitCompleteText(carry, delta) {
  const combined = carry + delta
  if (combined.length === 0) return { complete: "", carry: "" }
  const last = combined.charCodeAt(combined.length - 1)
  // Wait for the next chunk before counting an unfinished surrogate pair.
  if (isHighSurrogate(last)) {
    return { complete: combined.slice(0, -1), carry: combined.slice(-1) }
  }
  return { complete: combined, carry: "" }
}

function chunkText(chunk) {
  if (chunk.type === "text-delta" || chunk.type === "reasoning-delta") {
    return typeof chunk.text === "string" ? chunk.text : ""
  }
  if (chunk.type === "tool-call-delta") {
    return typeof chunk.argumentsDelta === "string" ? chunk.argumentsDelta : ""
  }
  return ""
}

function usageRecord(chunk) {
  if (chunk.type !== "usage") return null
  return {
    inputTokens: chunk.usage.inputTokens,
    outputTokens: chunk.usage.outputTokens,
    cacheReadTokens: chunk.usage.cacheReadTokens ?? null,
    cacheWriteTokens: chunk.usage.cacheWriteTokens ?? null,
    reasoningTokens: chunk.usage.reasoningTokens ?? null,
  }
}

function terminalFacts(reason) {
  const kind = reason?.kind ?? "unknown"
  const failure = reason?.failure
  return {
    finish_reason: kind,
    failure_code:
      kind === "error" || kind === "aborted"
        ? String(failure?.code ?? "UNKNOWN")
        : null,
    failure: failure
      ? {
          message: String(failure.message ?? ""),
          code: failure.code ?? null,
          status: failure.status ?? null,
          requestId: failure.requestId ?? null,
        }
      : null,
  }
}

export class NativeSessionIndex {
  constructor() {
    this.byType = new Map()
    this.messageSeqs = []
    this.logLength = 0
    this.pendingRetry = null
    this.lastRetryStarted = null
  }

  observe(event) {
    const seq = Number(event.seq)
    if (!Number.isSafeInteger(seq) || seq < 0) return
    this.logLength = Math.max(this.logLength, seq + 1)
    this.byType.set(String(event.type), { native_seq: seq, value: event.data })
    if (String(event.type).endsWith("/message") || String(event.type) === "user/message") {
      if (!this.messageSeqs.includes(seq)) this.messageSeqs.push(seq)
    }
  }

  latest(type) {
    return this.byType.get(type) ?? null
  }

  messageRefs() {
    return { native_seqs: [...this.messageSeqs] }
  }
}

export class DshGenerationObserver {
  constructor({
    checkpointIntervalBytes = 128,
    writeRecord,
    now = Date.now,
    measureTokens = null,
    onAttempt = null,
  } = {}) {
    if (!Number.isSafeInteger(checkpointIntervalBytes) || checkpointIntervalBytes < 1) {
      throw new TypeError("checkpointIntervalBytes must be a positive integer")
    }
    if (typeof writeRecord !== "function") {
      throw new TypeError("writeRecord must be a function")
    }
    if (measureTokens !== null && typeof measureTokens !== "function") {
      throw new TypeError("measureTokens must be a function or null")
    }
    this.checkpointIntervalBytes = checkpointIntervalBytes
    this.writeRecord = writeRecord
    this.now = now
    this.measureTokens = measureTokens
    this.onAttempt = onAttempt
    this.sessions = new Map()
  }

  stateFor(sessionId) {
    let state = this.sessions.get(sessionId)
    if (state === undefined) {
      state = {
        native: new NativeSessionIndex(),
        calls: new Map(),
        session: null,
      }
      this.sessions.set(sessionId, state)
    }
    return state
  }

  disposeSession(sessionId) {
    this.sessions.delete(sessionId)
  }

  observeNativeEvent(sessionId, event, session = null) {
    const state = this.stateFor(sessionId)
    if (session !== null) state.session = session
    state.native.observe(event)
    const type = String(event.type)
    if (type === NATIVE_RETRY_SCHEDULED) this.writeRetryScheduled(sessionId, event)
    else if (type === NATIVE_RETRY_STARTED) this.writeRetryStarted(sessionId, event)
  }

  callForStep(sessionId, turn, step) {
    return this.stateFor(sessionId).calls.get(`${turn}:${step}`) ?? null
  }

  writeRetryScheduled(sessionId, event) {
    const data = event.data ?? {}
    const call = this.callForStep(sessionId, data.turn, data.step)
    if (call === null) return
    const failure = data.failure ?? {}
    this.writeRecord(sessionId, {
      schema_version: OBSERVER_V3_SCHEMA_VERSION,
      record_type: "retry/scheduled",
      time: this.now(),
      session_id: sessionId,
      turn: data.turn,
      step: data.step,
      logical_call_id: call.logicalCallId,
      native_seq: event.seq,
      retry_id: data.retryId ?? null,
      retry_ordinal: data.retry ?? null,
      failed_attempt_id: call.lastAttemptId,
      provider: data.provider ?? null,
      mode: data.mode ?? null,
      policy_key: data.policyKey ?? null,
      delay_ms: data.delayMs ?? null,
      failure_code: failure.code ?? null,
      failure_status: failure.status ?? null,
      call_elapsed_ms: this.now() - call.startedAt,
      native_data: data,
    })
  }

  writeRetryStarted(sessionId, event) {
    const data = event.data ?? {}
    const call = this.callForStep(sessionId, data.turn, data.step)
    if (call === null) return
    const scheduled = this.stateFor(sessionId).native.latest(NATIVE_RETRY_SCHEDULED)
    this.writeRecord(sessionId, {
      schema_version: OBSERVER_V3_SCHEMA_VERSION,
      record_type: "retry/started",
      time: this.now(),
      session_id: sessionId,
      turn: data.turn,
      step: data.step,
      logical_call_id: call.logicalCallId,
      native_seq: event.seq,
      scheduled_native_seq: scheduled ? scheduled.native_seq : null,
      retry_id: data.retryId ?? null,
      retry_ordinal: data.retry ?? null,
      failed_attempt_id: call.lastAttemptId,
      call_elapsed_ms: this.now() - call.startedAt,
      native_data: data,
    })
  }

  sessionRecord(sessionId, { dshRelease = null, dshRevision = null } = {}) {
    return {
      schema_version: OBSERVER_V3_SCHEMA_VERSION,
      record_type: "observer/session",
      time: this.now(),
      session_id: sessionId,
      plugin_version: OBSERVER_V3_PLUGIN_VERSION,
      dsh_release: dshRelease,
      dsh_revision: dshRevision,
      dsh_llm_compatibility: DSH_LLM_COMPATIBILITY,
      observer_durability_contract_id: OBSERVER_DURABILITY_CONTRACT_ID,
      checkpoint_interval_bytes: this.checkpointIntervalBytes,
      checkpoint_policy_id: generationCheckpointPolicyId(this.checkpointIntervalBytes),
      observer_unit_contract_id: OBSERVER_UNIT_CONTRACT_ID,
      generation_unit_contract_id: GENERATION_UNIT_CONTRACT_ID,
      observer_tokenizer_id: OBSERVER_TOKENIZER_ID,
      observer_normalization_id: OBSERVER_NORMALIZATION_ID,
      observer_carry_policy_id: OBSERVER_CARRY_POLICY_ID,
      observer_boundary_policy_id: OBSERVER_BOUNDARY_POLICY_ID,
      attempt_boundary_contract_id: ATTEMPT_BOUNDARY_CONTRACT_ID,
      generation_channels: [...GENERATION_CHANNELS],
      capability_mask: {
        text: true,
        reasoning: true,
        tool_args: true,
        ordinal: true,
        elapsed: true,
        effective_request: true,
        request_context: true,
        token_meter: true,
        native_retry: true,
      },
      provider_wire_attempt: "unknown",
    }
  }

  requestProjection(request, native) {
    return {
      provider: request.provider ?? null,
      model: request.model ?? null,
      sessionId: request.sessionId ?? null,
      reasoningEffort: request.reasoningEffort ?? null,
      temperature: request.temperature ?? null,
      maxTokens: request.maxTokens ?? null,
      stop: Array.isArray(request.stop) ? [...request.stop] : [],
      purpose: request.purpose ?? null,
      message_count: Array.isArray(request.messages) ? request.messages.length : 0,
      tool_count: Array.isArray(request.tools) ? request.tools.length : 0,
      block_count: Array.isArray(request.messages)
        ? request.messages.reduce(
            (total, message) =>
              total + (Array.isArray(message?.content) ? message.content.length : 0),
            0,
          )
        : 0,
      messages_ref: native.messageRefs(),
    }
  }

  beginCall(request, step) {
    const sessionId = String(request.sessionId ?? "")
    if (!sessionId) throw new TypeError("agent-loop request is missing sessionId")
    const state = this.stateFor(sessionId)
    const key = `${step.turn}:${step.step}`
    let call = state.calls.get(key)
    if (call !== undefined) return call

    const native = state.native
    const logicalCallId = `${sessionId}:turn-${step.turn}:step-${step.step}`
    call = {
      key,
      logicalCallId,
      turn: step.turn,
      step: step.step,
      attemptOrdinal: 0,
      lastAttemptId: null,
      startedAt: this.now(),
    }
    state.calls.set(key, call)
    call.effectiveRequest = effectiveRequestOf(
      this.requestProjection(request, native),
      native.latest(NATIVE_REQUEST_HEADER),
      native.latest(NATIVE_REQUEST_CONTEXT),
    )
    this.writeRecord(sessionId, {
      schema_version: OBSERVER_V3_SCHEMA_VERSION,
      record_type: "call/start",
      time: call.startedAt,
      session_id: sessionId,
      turn: call.turn,
      step: call.step,
      logical_call_id: logicalCallId,
      native_format_revision: NATIVE_FORMAT_REVISION,
      effective_request_projection: this.requestProjection(request, native),
      request_header: native.latest(NATIVE_REQUEST_HEADER),
      request_context: native.latest(NATIVE_REQUEST_CONTEXT),
      request_feature_mask: requestFeatureMask(request),
      token_meter: this.tokenMeter(sessionId),
      provider_attempt_status: "llm_stream_entered",
    })
    return call
  }

  tokenMeter(sessionId) {
    if (this.measureTokens === null) {
      throw new TypeError(
        "the v3 observer requires the harness token meter to record a request",
      )
    }
    const state = this.stateFor(sessionId)
    const header = state.native.latest(NATIVE_REQUEST_HEADER)
    const meter = this.measureTokens(state.session, header ? header.value : undefined)
    return structuredClone(meter)
  }

  beginAttempt(request, step) {
    const sessionId = String(request.sessionId ?? "")
    const state = this.stateFor(sessionId)
    const native = state.native
    const call = this.beginCall(request, step)
    call.attemptOrdinal += 1
    const attemptId = `${call.logicalCallId}:attempt-${call.attemptOrdinal}`
    const retryOfAttemptId = call.lastAttemptId
    call.lastAttemptId = attemptId

    const scheduled = native.latest(NATIVE_RETRY_SCHEDULED)
    const started = native.latest(NATIVE_RETRY_STARTED)
    const isRetry = call.attemptOrdinal > 1
    const facts = {
      sessionId,
      turn: call.turn,
      step: call.step,
      logicalCallId: call.logicalCallId,
      attemptId,
      attemptOrdinal: call.attemptOrdinal,
      callStartedAt: call.startedAt,
      startedAt: this.now(),
    }
    this.writeRecord(sessionId, {
      schema_version: OBSERVER_V3_SCHEMA_VERSION,
      record_type: "attempt/start",
      time: facts.startedAt,
      session_id: sessionId,
      turn: call.turn,
      step: call.step,
      logical_call_id: call.logicalCallId,
      attempt_id: attemptId,
      attempt_ordinal: call.attemptOrdinal,
      retry_of_attempt_id: retryOfAttemptId,
      // The native retry payload uses camelCase; the exported record uses snake_case.
      retry_id: isRetry && scheduled ? (scheduled.value?.retryId ?? null) : null,
      retry_ordinal:
        isRetry && scheduled
          ? (scheduled.value?.retry ?? call.attemptOrdinal - 1)
          : null,
      retry_scheduled_native_seq: isRetry && scheduled ? scheduled.native_seq : null,
      retry_started_native_seq: isRetry && started ? started.native_seq : null,
      provider: String(request.provider),
      model: String(request.model),
      native_format_revision: NATIVE_FORMAT_REVISION,
      effective_request_projection: this.requestProjection(request, native),
      request_header: native.latest(NATIVE_REQUEST_HEADER),
      request_context: native.latest(NATIVE_REQUEST_CONTEXT),
      request_changed_from_initial_call:
        effectiveRequestOf(
          this.requestProjection(request, native),
          native.latest(NATIVE_REQUEST_HEADER),
          native.latest(NATIVE_REQUEST_CONTEXT),
        ) !== call.effectiveRequest,
      request_feature_mask: requestFeatureMask(request),
      token_meter: this.tokenMeter(sessionId),
    })
    return facts
  }

  observe(request, stream, step) {
    const observer = this
    return (async function* observedGeneration() {
      const facts = observer.beginAttempt(request, step)
      observer.onAttempt?.(request, facts)
      const state = observer.stateFor(facts.sessionId)
      const carry = { text: "", reasoning: "", tool_arguments: "" }
      const bytes = zeroChannels()
      const seen = falseChannels()
      let generationBytes = 0
      let checkpointBytes = 0
      let checkpointOrdinal = 0
      let streamEventOrdinal = 0
      let usage = null
      let ended = false

      const appendChannel = (channel, text) => {
        const split = splitCompleteText(carry[channel], text)
        carry[channel] = split.carry
        const normalized = split.complete.toWellFormed()
        const added = Buffer.byteLength(normalized)
        bytes[channel] += added
        generationBytes += added
        if (added > 0) seen[channel] = true
      }

      const flushCarry = () => {
        for (const channel of GENERATION_CHANNELS) {
          if (!carry[channel]) continue
          const normalized = carry[channel].toWellFormed()
          const added = Buffer.byteLength(normalized)
          bytes[channel] += added
          generationBytes += added
          carry[channel] = ""
        }
      }

      const writeEnd = (finishReason, observerFinalized) => {
        if (ended) return
        ended = true
        flushCarry()
        const terminal = terminalFacts(finishReason)
        observer.writeRecord(facts.sessionId, {
          schema_version: OBSERVER_V3_SCHEMA_VERSION,
          record_type: "attempt/end",
          time: observer.now(),
          session_id: facts.sessionId,
          turn: facts.turn,
          step: facts.step,
          logical_call_id: facts.logicalCallId,
          attempt_id: facts.attemptId,
          attempt_ordinal: facts.attemptOrdinal,
          final_stream_event_ordinal: streamEventOrdinal,
          last_native_chunk_seq: state.native.logLength - 1,
          channel_bytes: { ...bytes },
          generation_bytes: generationBytes,
          attempt_elapsed_ms: observer.now() - facts.startedAt,
          call_elapsed_ms: observer.now() - facts.callStartedAt,
          finish_reason: terminal.finish_reason,
          failure_code: terminal.failure_code,
          failure: terminal.failure,
          last_provider_usage: usage,
          observer_finalized: observerFinalized,
          observed_channel_mask: { ...seen },
        })
      }

      const commitChunk = (chunk) => {
        // Count usage and finish chunks too, keeping this ordinal aligned with the session log.
        streamEventOrdinal += 1
        const channel = CHANNEL_BY_CHUNK_TYPE[chunk.type]
        if (channel !== undefined) {
          appendChannel(channel, chunkText(chunk))
          if (generationBytes - checkpointBytes >= observer.checkpointIntervalBytes) {
            checkpointOrdinal += 1
            checkpointBytes = generationBytes
            observer.writeRecord(facts.sessionId, {
              schema_version: OBSERVER_V3_SCHEMA_VERSION,
              record_type: "attempt/checkpoint",
              time: observer.now(),
              session_id: facts.sessionId,
              turn: facts.turn,
              step: facts.step,
              logical_call_id: facts.logicalCallId,
              attempt_id: facts.attemptId,
              attempt_ordinal: facts.attemptOrdinal,
              checkpoint_ordinal: checkpointOrdinal,
              stream_event_ordinal: streamEventOrdinal,
              native_chunk_seq: state.native.logLength - 1,
              channel_bytes: { ...bytes },
              generation_bytes: generationBytes,
              attempt_elapsed_ms: observer.now() - facts.startedAt,
              call_elapsed_ms: observer.now() - facts.callStartedAt,
              seen_channel_mask: { ...seen },
              probe_checkpoint_available: false,
            })
          }
        } else if (chunk.type === "usage") {
          usage = usageRecord(chunk)
        } else if (chunk.type === "finish") {
          writeEnd(chunk.reason, true)
        }
      }

      try {
        for await (const chunk of stream) {
          commitChunk(chunk)
          yield chunk
        }
        writeEnd({ kind: "missing-finish" }, false)
      } catch (error) {
        writeEnd({ kind: "error", failure: { code: "STREAM_EXCEPTION", message: String(error) } }, false)
        throw error
      }
    })()
  }
}

export const NATIVE_FORMAT_REVISION = "dsh-session-v0-rc.2"

export const REQUEST_FEATURE_FIELDS = [
  "reasoningEffort",
  "temperature",
  "maxTokens",
  "stop",
  "purpose",
]

export function effectiveRequestOf(projection, headerSnapshot, contextSnapshot) {
  return JSON.stringify([
    projection,
    headerSnapshot?.value?.header ?? null,
    contextSnapshot?.value ?? null,
  ])
}

function requestFeatureMask(request) {
  const mask = {}
  for (const field of REQUEST_FEATURE_FIELDS) {
    mask[field] = request[field] !== undefined && request[field] !== null
  }
  return mask
}
