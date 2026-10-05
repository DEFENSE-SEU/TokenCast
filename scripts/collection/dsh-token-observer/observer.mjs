import { isDeepStrictEqual } from "node:util"

export const OBSERVER_SCHEMA_VERSION = 2
export const OBSERVER_PLUGIN_VERSION = "0.4.0"
export const DSH_LLM_COMPATIBILITY = ">=0.1.1-rc.2 <0.2.0"
export const OBSERVER_DURABILITY_CONTRACT_ID =
  "observer-jsonl-fsync-on-session-flush-v1"
export const OBSERVER_UNIT_CONTRACT_ID = "observer:utf8-bytes-v1"
export const OBSERVER_TOKENIZER_ID = "utf8-byte-256"
export const OBSERVER_NORMALIZATION_ID = "utf16-to-well-formed-v1"
export const OBSERVER_CARRY_POLICY_ID = "utf16-high-surrogate-carry-v1"
export const OBSERVER_BOUNDARY_POLICY_ID = "dsh-text-delta-chunk-v1"
export const GENERATION_PROBE_BRIDGE = Symbol.for("token-prediction.dsh-generation-probe.v1")

export function generationProbeBridgeRegistry() {
  globalThis[GENERATION_PROBE_BRIDGE] ??= new WeakMap()
  return globalThis[GENERATION_PROBE_BRIDGE]
}

const REQUEST_FIELDS = [
  "provider",
  "model",
  "reasoningEffort",
  "messages",
  "system",
  "tools",
  "temperature",
  "maxTokens",
  "stop",
  "purpose",
]

export function requestSnapshot(request) {
  const snapshot = {}
  for (const field of REQUEST_FIELDS) {
    if (request[field] !== undefined) snapshot[field] = request[field]
  }
  return snapshot
}

export function providerUnitContractId(provider, model) {
  return `provider:dsh-outputTokens-v1:${encodeURIComponent(provider)}:${encodeURIComponent(model)}`
}

export function checkpointPolicyId(checkpointIntervalBytes, countToolArguments = false) {
  // Include the counted channels in the policy ID when comparing captures.
  const scope = countToolArguments ? "output-chunk" : "text-chunk"
  return `dsh-${scope}-at-least-${checkpointIntervalBytes}-utf8-bytes-v1`
}

function isHighSurrogate(code) {
  return code >= 0xd800 && code <= 0xdbff
}

function splitCompleteText(carry, delta) {
  const combined = carry + delta
  if (combined.length === 0) return { complete: "", carry: "" }
  const last = combined.charCodeAt(combined.length - 1)
  // Keep a trailing high surrogate until the next chunk can complete the character.
  if (isHighSurrogate(last)) {
    return {
      complete: combined.slice(0, -1),
      carry: combined.slice(-1),
    }
  }
  return { complete: combined, carry: "" }
}

function requestShape(request) {
  return {
    message_count: Array.isArray(request.messages) ? request.messages.length : 0,
    tool_count: Array.isArray(request.tools) ? request.tools.length : 0,
    system_utf8_bytes: typeof request.system === "string" ? Buffer.byteLength(request.system) : 0,
  }
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
  return {
    finish_reason: kind,
    failure_code:
      kind === "error" || kind === "aborted"
        ? String(reason.failure?.code ?? "UNKNOWN")
        : null,
  }
}

function validateStep(step) {
  if (!Number.isSafeInteger(step.turn) || step.turn < 0) {
    throw new TypeError("DSH observer requires a non-negative integer turn")
  }
  if (!Number.isSafeInteger(step.step) || step.step < 0) {
    throw new TypeError("DSH observer requires a non-negative integer step")
  }
}

export class DshStreamObserver {
  constructor({
    checkpointIntervalBytes = 128,
    countToolArguments = false,
    writeRecord,
    now = Date.now,
    onAttempt = null,
  } = {}) {
    if (!Number.isSafeInteger(checkpointIntervalBytes) || checkpointIntervalBytes <= 0) {
      throw new TypeError("checkpointIntervalBytes must be a positive integer")
    }
    if (typeof writeRecord !== "function") {
      throw new TypeError("writeRecord must be a function")
    }
    this.checkpointIntervalBytes = checkpointIntervalBytes
    this.countToolArguments = Boolean(countToolArguments)
    this.writeRecord = writeRecord
    this.now = now
    this.onAttempt = onAttempt
    this.sessions = new Map()
  }

  sessionRecord(sessionId) {
    return {
      schema_version: OBSERVER_SCHEMA_VERSION,
      record_type: "observer/session",
      time: this.now(),
      session_id: sessionId,
      plugin_version: OBSERVER_PLUGIN_VERSION,
      dsh_llm_compatibility: DSH_LLM_COMPATIBILITY,
      observer_durability_contract_id: OBSERVER_DURABILITY_CONTRACT_ID,
      checkpoint_interval_bytes: this.checkpointIntervalBytes,
      checkpoint_policy_id: checkpointPolicyId(
        this.checkpointIntervalBytes,
        this.countToolArguments,
      ),
      observer_unit_contract_id: OBSERVER_UNIT_CONTRACT_ID,
      observer_tokenizer_id: OBSERVER_TOKENIZER_ID,
      observer_normalization_id: OBSERVER_NORMALIZATION_ID,
      observer_carry_policy_id: OBSERVER_CARRY_POLICY_ID,
      observer_boundary_policy_id: OBSERVER_BOUNDARY_POLICY_ID,
    }
  }

  stateFor(sessionId) {
    let state = this.sessions.get(sessionId)
    if (state === undefined) {
      state = { steps: new Map() }
      this.sessions.set(sessionId, state)
      this.writeRecord(sessionId, this.sessionRecord(sessionId))
    }
    return state
  }

  disposeSession(sessionId) {
    this.sessions.delete(sessionId)
  }

  beginAttempt(request, step) {
    validateStep(step)
    const sessionId = String(request.sessionId ?? "")
    if (!sessionId) throw new TypeError("agent-loop request is missing sessionId")
    const state = this.stateFor(sessionId)
    const stepKey = `${step.turn}:${step.step}`
    const logicalCallId = `${sessionId}:turn-${step.turn}:step-${step.step}`
    const requestId = `${logicalCallId}:request`
    const snapshot = requestSnapshot(request)
    let stepState = state.steps.get(stepKey)
    if (stepState === undefined) {
      stepState = {
        requestId,
        requestSnapshot: structuredClone(snapshot),
        attemptOrdinal: 0,
        lastAttemptId: null,
      }
      state.steps.set(stepKey, stepState)
    } else if (!isDeepStrictEqual(stepState.requestSnapshot, snapshot)) {
      throw new TypeError("provider retry changed the logical Call request")
    }
    stepState.attemptOrdinal += 1
    const attemptId = `${logicalCallId}:attempt-${stepState.attemptOrdinal}`
    const retryOfAttemptId = stepState.lastAttemptId
    stepState.lastAttemptId = attemptId
    const provider = String(request.provider)
    const model = String(request.model)
    const facts = {
      sessionId,
      turn: step.turn,
      step: step.step,
      logicalCallId,
      requestId: stepState.requestId,
      attemptId,
      attemptOrdinal: stepState.attemptOrdinal,
      retryOfAttemptId,
      provider,
      model,
      reasoningEffort: request.reasoningEffort ?? null,
      providerUnitContractId: providerUnitContractId(provider, model),
    }
    this.writeRecord(sessionId, {
      schema_version: OBSERVER_SCHEMA_VERSION,
      record_type: "attempt/start",
      time: this.now(),
      session_id: sessionId,
      turn: facts.turn,
      step: facts.step,
      logical_call_id: facts.logicalCallId,
      request_id: facts.requestId,
      attempt_id: facts.attemptId,
      attempt_ordinal: facts.attemptOrdinal,
      retry_of_attempt_id: facts.retryOfAttemptId,
      provider: facts.provider,
      model: facts.model,
      reasoning_effort: facts.reasoningEffort,
      provider_token_unit_contract_id: facts.providerUnitContractId,
      request_shape: requestShape(request),
    })
    return facts
  }

  observe(request, stream, step) {
    const observer = this
    return (async function* observedStream() {
      const facts = observer.beginAttempt(request, step)
      observer.onAttempt?.(request, facts)
      let carry = ""
      let pendingText = ""
      let visibleBytes = 0
      let checkpointBytes = 0
      let checkpointOrdinal = 0
      let usage = null
      let ended = false

      const appendVisible = (text) => {
        const split = splitCompleteText(carry, text)
        carry = split.carry
        const normalized = split.complete.toWellFormed()
        pendingText += normalized
        visibleBytes += Buffer.byteLength(normalized)
      }

      const commitChunk = (chunk) => {
        const counts =
          chunk.type === "text-delta"
          || (observer.countToolArguments && chunk.type === "tool-call-delta")
        if (counts) {
          appendVisible(chunk.type === "text-delta" ? chunk.text : chunk.argumentsDelta ?? "")
          if (visibleBytes - checkpointBytes >= observer.checkpointIntervalBytes) {
            checkpointOrdinal += 1
            checkpointBytes = visibleBytes
            const checkpoint = {
              schema_version: OBSERVER_SCHEMA_VERSION,
              record_type: "attempt/checkpoint",
              time: observer.now(),
              session_id: facts.sessionId,
              turn: facts.turn,
              step: facts.step,
              logical_call_id: facts.logicalCallId,
              attempt_id: facts.attemptId,
              attempt_ordinal: facts.attemptOrdinal,
              checkpoint_ordinal: checkpointOrdinal,
              observer_committed_tokens: visibleBytes,
              observer_unit_contract_id: OBSERVER_UNIT_CONTRACT_ID,
              visible_text_delta: pendingText,
            }
            observer.writeRecord(facts.sessionId, checkpoint)
            const bridge = generationProbeBridgeRegistry().get(chunk)
            if (bridge) {
              bridge.checkpoints.push({
                session_id: checkpoint.session_id,
                attempt_id: checkpoint.attempt_id,
                checkpoint_ordinal: checkpoint.checkpoint_ordinal,
                observer_committed_tokens: checkpoint.observer_committed_tokens,
              })
            }
            pendingText = ""
          }
        } else if (chunk.type === "usage") {
          usage = usageRecord(chunk)
        } else if (chunk.type === "finish") {
          writeEnd(chunk.reason, true)
        }
      }

      const writeEnd = (finishReason, observerFinalized) => {
        if (ended) return
        ended = true
        if (carry) {
          const normalized = carry.toWellFormed()
          pendingText += normalized
          visibleBytes += Buffer.byteLength(normalized)
          carry = ""
        }
        const terminal = terminalFacts(finishReason)
        observer.writeRecord(facts.sessionId, {
          schema_version: OBSERVER_SCHEMA_VERSION,
          record_type: "attempt/end",
          time: observer.now(),
          session_id: facts.sessionId,
          turn: facts.turn,
          step: facts.step,
          logical_call_id: facts.logicalCallId,
          attempt_id: facts.attemptId,
          attempt_ordinal: facts.attemptOrdinal,
          provider_token_unit_contract_id: facts.providerUnitContractId,
          finish_reason: terminal.finish_reason,
          failure_code: terminal.failure_code,
          usage,
          observer_visible_output_tokens: visibleBytes,
          observer_finalized: observerFinalized,
          visible_text_tail: pendingText,
        })
      }

      let phase = "source"
      try {
        for await (const chunk of stream) {
          phase = "consumer"
          try {
            yield chunk
          } catch (error) {
            writeEnd({ kind: "consumer-exception" }, false)
            throw error
          }
          // Count the chunk after downstream processing resumes the generator.
          phase = "observer"
          commitChunk(chunk)
          phase = "source"
        }
        if (!ended) writeEnd({ kind: "missing-finish" }, false)
      } catch (error) {
        if (!ended) {
          const kind = phase === "source" ? "stream-exception" : "observer-exception"
          writeEnd({ kind }, false)
        }
        throw error
      } finally {
        if (!ended) writeEnd({ kind: "consumer-cancelled" }, false)
      }
    })()
  }
}

export function createStreamMiddleware({
  observer,
  isAgentLoopRequest,
  activeStep,
  onMissingStep = () => {},
}) {
  return (options, next) => {
    if (!isAgentLoopRequest(options)) return next()
    const sessionId = String(options.sessionId ?? "")
    const step = activeStep(sessionId)
    if (step === undefined) {
      onMissingStep(sessionId)
      return next()
    }
    const downstream = observer.onAttempt
      ? (async function* deferredDispatch() { yield* next() })()
      : next()
    return observer.observe(options, downstream, step)
  }
}
