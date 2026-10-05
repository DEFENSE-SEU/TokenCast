import { closeSync, fsyncSync, mkdirSync, openSync, writeSync } from "node:fs"
import { dirname, isAbsolute } from "node:path"

function nonnegative(value, label) {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error(`Budget shadow: invalid ${label}`)
  return value
}

function requiredText(value, label) {
  if (typeof value !== "string" || !value.trim()) throw new Error(`Budget shadow: missing ${label}`)
  return value
}

function usageOf(chunk) {
  if (chunk?.type !== "usage") return null
  const usage = chunk.usage
  if (usage === null || typeof usage !== "object") throw new Error("Budget shadow: missing provider usage")
  const input = nonnegative(usage.inputTokens, "inputTokens")
  const output = nonnegative(usage.outputTokens, "outputTokens")
  const cacheRead = nonnegative(usage.cacheReadTokens ?? 0, "cacheReadTokens")
  const cacheWrite = nonnegative(usage.cacheWriteTokens ?? 0, "cacheWriteTokens")
  const reasoning = nonnegative(usage.reasoningTokens ?? 0, "reasoningTokens")
  if (reasoning > output) throw new Error("Budget shadow: reasoning exceeds output")
  return {
    input_tokens: input,
    output_tokens: output,
    cache_read_tokens: cacheRead,
    cache_write_tokens: cacheWrite,
    reasoning_output_tokens: reasoning,
    provider_accounted_tokens: input + output + cacheRead + cacheWrite,
  }
}

export function createBudgetShadow({
  path, sourceId, taskFamily, runId, sessionId, provider, model,
  budgetTokens, marginTokens = 0, measure, accountedBefore, currentStep,
  now = () => new Date().toISOString(),
}) {
  if (typeof path !== "string" || !isAbsolute(path)) throw new Error("Budget shadow: absolute path required")
  for (const [label, value] of Object.entries({ sourceId, taskFamily, runId, sessionId, provider, model })) {
    requiredText(value, label)
  }
  nonnegative(budgetTokens, "budgetTokens")
  nonnegative(marginTokens, "marginTokens")
  if (typeof measure !== "function" || typeof accountedBefore !== "function"
    || typeof currentStep !== "function") throw new Error("Budget shadow: callbacks required")
  mkdirSync(dirname(path), { recursive: true })
  const descriptor = openSync(path, "wx")
  let closed = false
  const ordinals = new Map()
  function append(record) {
    if (closed) throw new Error("Budget shadow: writer closed")
    const line = `${JSON.stringify({ schema_version: "dsh-budget-shadow-v1", ...record })}\n`
    if (writeSync(descriptor, line) !== Buffer.byteLength(line)) {
      throw new Error("Budget shadow: incomplete record write")
    }
    fsyncSync(descriptor)
  }
  try {
    append({ record_type: "scope", at: now(), source_id: sourceId,
      task_family: taskFamily, run_id: runId, session_id: sessionId,
      provider, model, budget_tokens: budgetTokens, margin_tokens: marginTokens,
      decision_mode: "shadow_only" })
  } catch (error) {
    closeSync(descriptor)
    throw error
  }

  function middleware(request, next) {
    if (String(request?.sessionId ?? "") !== sessionId) return next()
    if (request.provider !== provider || request.model !== model) {
      append({ record_type: "route_mismatch", at: now(),
        requested_provider: request.provider ?? null, requested_model: request.model ?? null,
        decision_status: "unavailable", policy_action: "pass_through" })
      return next()
    }
    return (async function* observedAttempt() {
      let step = null
      try { step = currentStep() } catch {   }
      const validStep = step && Number.isSafeInteger(step.turn) && Number.isSafeInteger(step.step)
      const turn = validStep ? step.turn : null
      const stepNumber = validStep ? step.step : null
      const key = `${turn}:${stepNumber}`
      const attemptOrdinal = (ordinals.get(key) ?? 0) + 1
      ordinals.set(key, attemptOrdinal)
      let measurement = null
      let estimate = null
      try {
        measurement = measure()
        if (measurement?.baseline?.kind !== "none") {
          estimate = nonnegative(measurement?.totalTokens, "meter totalTokens")
        }
      } catch {   }
      let prior = null
      try {
        const value = accountedBefore()
        if (value !== null) prior = nonnegative(value, "settled tokens before dispatch")
      } catch {   }
      const projected = !validStep || prior === null || estimate === null
        ? null : prior + Math.max(0, estimate - marginTokens)
      const attemptId = `${sessionId}:turn-${turn}:step-${stepNumber}:attempt-${attemptOrdinal}`
      // Record the decision before dispatch; shadow mode always lets the request proceed.
      append({ record_type: "pre_dispatch", at: now(), attempt_id: attemptId,
        turn, step: stepNumber, attempt_ordinal: attemptOrdinal,
        retry: attemptOrdinal > 1, provider, model,
        dsh_meter_estimate_tokens: estimate,
        dsh_meter_basis: measurement?.baseline?.kind ?? null,
        settled_provider_accounted_tokens_before: prior,
        projected_provider_accounted_tokens: projected,
        would_stop: projected === null ? null : projected > budgetTokens,
        decision_status: projected === null ? "unavailable" : "available",
        policy_action: "pass_through" })
      let usage = null
      let usageStatus = "missing"
      let finish = null
      let streamStatus = "incomplete"
      try {
        for await (const chunk of next()) {
          if (chunk?.type === "usage") {
            try {
              usage = usageOf(chunk)
              usageStatus = "provider_reported"
            } catch {
              usage = null
              usageStatus = "invalid"
            }
          }
          if (chunk?.type === "finish") finish = chunk.reason?.kind ?? null
          yield chunk
        }
        streamStatus = "complete"
      } catch (error) {
        streamStatus = "stream_error"
        throw error
      } finally {
        append({ record_type: "attempt_end", at: now(), attempt_id: attemptId,
          stream_status: streamStatus, finish_reason: finish,
          usage_status: usageStatus,
          usage, yuan_spend: null })
      }
    })()
  }

  return {
    middleware,
    close() {
      if (closed) return
      closed = true
      closeSync(descriptor)
    },
  }
}
