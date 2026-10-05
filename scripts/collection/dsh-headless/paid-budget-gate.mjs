import { createHash } from "node:crypto"
import { closeSync, existsSync, fsyncSync, mkdirSync, openSync, readFileSync, unlinkSync, writeSync } from "node:fs"
import { dirname, isAbsolute, resolve } from "node:path"

const ROUTE = "https://openai.sufy.com/v1/chat/completions"
const MODEL = "openai/gpt-5.4"
const YUAN_LIMIT = 90
const MAX_INPUT_BOUND = 40_000
const MAX_OUTPUT_TOKENS = 1_280
const MAX_REQUESTS = 97
const INPUT_USD_PER_M = 2.5
const OUTPUT_USD_PER_M = 15
const YUAN_PER_USD = 8
const PRICE_RESERVE_FACTOR = 5
const FRAMING_TOKEN_RESERVE = 4_096
const YUAN_UNITS = 10_000

function chargeUnits(inputTokens, outputTokens) {
  return inputTokens + 6 * outputTokens
}

function diagnosticIdentifier(value) {
  return typeof value === "string" && /^[A-Za-z][A-Za-z0-9_]{0,63}$/.test(value)
    ? value : undefined
}

function transportDiagnostic(error) {
  const diagnostic = {}
  try {
    const name = diagnosticIdentifier(error?.name)
    if (name) diagnostic.error_name = name
  } catch {   }
  try {
    const cause = error?.cause
    const name = diagnosticIdentifier(cause?.name)
    const code = diagnosticIdentifier(cause?.code)
    if (name) diagnostic.cause_name = name
    if (code) diagnostic.cause_code = code
  } catch {   }
  return diagnostic
}

function validUsage(usage, reserve) {
  return usage && typeof usage === "object" && !Array.isArray(usage)
    && Number.isSafeInteger(usage.prompt_tokens) && usage.prompt_tokens > 0
    && Number.isSafeInteger(usage.completion_tokens) && usage.completion_tokens >= 0
    && Number.isSafeInteger(usage.total_tokens)
    && usage.total_tokens === usage.prompt_tokens + usage.completion_tokens
    && usage.prompt_tokens <= reserve.input_bound
    && usage.completion_tokens <= reserve.max_completion_tokens
}

function requirePositiveInteger(value, label) {
  if (!Number.isSafeInteger(value) || value < 1) throw new Error(`Paid budget: invalid ${label}`)
  return value
}

function requestBodyText(input, init) {
  if (init !== undefined || !(input instanceof Request)) {
    throw new Error("Paid budget: reserve requires the final Request without overrides")
  }
  return input.clone().text()
}

function onlyKeys(value, allowed, label) {
  if (typeof value !== "object" || value === null || Array.isArray(value)
    || Object.keys(value).some((key) => !allowed.includes(key))) {
    throw new Error(`Paid budget: unsupported ${label} field`)
  }
}

function checkContent(messages) {
  for (const message of messages) {
    onlyKeys(message, ["role", "content", "tool_calls", "tool_call_id", "name", "reasoning_details"], "message")
    if (!["system", "developer", "user", "assistant", "tool"].includes(message.role)) {
      throw new Error("Paid budget: invalid message role")
    }
    if (message.tool_calls !== undefined && message.role !== "assistant") {
      throw new Error("Paid budget: tool calls require assistant role")
    }
    if (message.tool_call_id !== undefined && (message.role !== "tool"
      || typeof message.tool_call_id !== "string")) {
      throw new Error("Paid budget: invalid tool result")
    }
    if (message.name !== undefined && (message.role !== "tool" || typeof message.name !== "string")) {
      throw new Error("Paid budget: invalid tool result name")
    }
    if (message.reasoning_details !== undefined) {
      if (message.role !== "assistant" || !Array.isArray(message.reasoning_details)
        || !message.reasoning_details.every((detail) => {
          try {
            onlyKeys(detail, ["type", "id", "data"], "reasoning detail")
            return detail.type === "reasoning.encrypted"
              && typeof detail.id === "string" && typeof detail.data === "string"
          } catch { return false }
        })) {
        throw new Error("Paid budget: invalid reasoning details")
      }
    }
    if (message.tool_calls !== undefined) {
      if (!Array.isArray(message.tool_calls) || !message.tool_calls.every((call) => {
        try {
          onlyKeys(call, ["id", "type", "function"], "tool call")
          onlyKeys(call.function, ["name", "arguments"], "tool function")
          return call.type === "function" && typeof call.id === "string"
            && typeof call.function.name === "string"
            && typeof call.function.arguments === "string"
        } catch { return false }
      })) throw new Error("Paid budget: invalid tool calls")
    }
    const content = message.content
    if (typeof content === "string" || content === null && message.role === "assistant") continue
    if (!Array.isArray(content) || !content.every((part) => {
      try {
        onlyKeys(part, ["type", "text"], "message content")
        return part.type === "text" && typeof part.text === "string"
      } catch { return false }
    })) {
      throw new Error("Paid budget: non-text message content is not priced by this gate")
    }
  }
}

function checkTools(tools) {
  if (tools === undefined) return
  if (!Array.isArray(tools) || !tools.every((tool) => {
    try {
      onlyKeys(tool, ["type", "function"], "tool")
      onlyKeys(tool.function, ["name", "description", "parameters", "strict"], "tool definition")
      return tool.type === "function" && typeof tool.function.name === "string"
        && typeof tool.function.description === "string"
        && typeof tool.function.parameters === "object" && tool.function.parameters !== null
        && (tool.function.strict === undefined || typeof tool.function.strict === "boolean")
    } catch { return false }
  })) throw new Error("Paid budget: unsupported tool definition")
}

export function boundPaidRequest(url, bodyText) {
  if (url !== ROUTE) throw new Error("Paid budget: unexpected provider route")
  if (typeof bodyText !== "string") throw new Error("Paid budget: request body unavailable")
  let body
  try { body = JSON.parse(bodyText) } catch { throw new Error("Paid budget: invalid request JSON") }
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    throw new Error("Paid budget: invalid request object")
  }
  if (body.model !== MODEL || body.stream !== true || body.n !== undefined && body.n !== 1) {
    throw new Error("Paid budget: model, stream, or candidate count drift")
  }
  if (body.max_tokens !== undefined || body.max_output_tokens !== undefined
    || body.temperature !== undefined) {
    throw new Error("Paid budget: ambiguous output limit or unsupported temperature")
  }
  onlyKeys(body, ["model", "messages", "stream", "stream_options", "store",
    "max_completion_tokens", "tools", "tool_choice", "reasoning_effort", "n"], "request")
  if (body.reasoning_effort !== "low" || body.store !== undefined && body.store !== false) {
    throw new Error("Paid budget: reasoning effort or store policy drift")
  }
  if (body.stream_options !== undefined) {
    onlyKeys(body.stream_options, ["include_usage"], "stream options")
    if (body.stream_options.include_usage !== true) {
      throw new Error("Paid budget: streaming usage must be requested")
    }
  }
  if (body.tool_choice !== undefined) {
    if (typeof body.tool_choice === "string") {
      if (!["auto", "none", "required"].includes(body.tool_choice)) {
        throw new Error("Paid budget: invalid tool choice")
      }
    } else {
      onlyKeys(body.tool_choice, ["type", "function"], "tool choice")
      onlyKeys(body.tool_choice.function, ["name"], "tool choice function")
      if (body.tool_choice.type !== "function"
        || typeof body.tool_choice.function.name !== "string") {
        throw new Error("Paid budget: invalid tool choice")
      }
    }
  }
  const outputLimit = requirePositiveInteger(body.max_completion_tokens, "max_completion_tokens")
  if (outputLimit > MAX_OUTPUT_TOKENS) throw new Error("Paid budget: output limit exceeded")
  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    throw new Error("Paid budget: messages missing")
  }
  checkContent(body.messages)
  checkTools(body.tools)
  const requestBytes = Buffer.byteLength(bodyText, "utf8")
  const inputBound = requestBytes + FRAMING_TOKEN_RESERVE
  if (inputBound > MAX_INPUT_BOUND) throw new Error("Paid budget: input bound exceeded")
  const reservedYuan = PRICE_RESERVE_FACTOR * YUAN_PER_USD
    * (INPUT_USD_PER_M * inputBound + OUTPUT_USD_PER_M * outputLimit) / 1_000_000
  return {
    requestBytes, inputBound, outputLimit, reservedYuan,
    requestSha256: createHash("sha256").update(bodyText).digest("hex"),
  }
}

function readReservations(path) {
  if (!existsSync(path)) return { units: 0, grossUnits: 0, count: 0, attempts: new Map() }
  const text = readFileSync(path, "utf8")
  if (text && !text.endsWith("\n")) throw new Error("Paid budget: incomplete ledger; manual audit required")
  let units = 0
  let grossUnits = 0
  let count = 0
  const attempts = new Map()
  for (const line of text.split("\n")) {
    if (!line) continue
    let entry
    try { entry = JSON.parse(line) } catch { throw new Error("Paid budget: invalid ledger; manual audit required") }
    if (entry?.kind === "reserve") {
      if (entry.route !== ROUTE || entry.model !== MODEL
        || entry.attempt !== count + 1
        || !Number.isSafeInteger(entry.input_bound) || entry.input_bound < 1
        || entry.input_bound > MAX_INPUT_BOUND
        || !Number.isSafeInteger(entry.max_completion_tokens)
        || entry.max_completion_tokens < 1
        || entry.max_completion_tokens > MAX_OUTPUT_TOKENS
        || !Number.isFinite(entry.reserved_yuan) || entry.reserved_yuan <= 0) {
        throw new Error("Paid budget: invalid reservation; manual audit required")
      }
      const reserveUnits = chargeUnits(entry.input_bound, entry.max_completion_tokens)
      if (Math.abs(entry.reserved_yuan - reserveUnits / YUAN_UNITS) > 1e-9) {
        throw new Error("Paid budget: reservation arithmetic drift; manual audit required")
      }
      units += reserveUnits
      grossUnits += reserveUnits
      count += 1
      attempts.set(count, { reserve: entry, reserveUnits, http: null, transportError: false,
        sse: null, settled: null })
    } else if (entry?.kind === "http" || entry?.kind === "transport_error") {
      const state = attempts.get(entry.attempt)
      if (!state || state.http !== null || state.transportError) {
        throw new Error("Paid budget: duplicate or orphan transport result; manual audit required")
      }
      if (entry.kind === "http") {
        if (!Number.isSafeInteger(entry.status) || entry.status < 100 || entry.status > 599) {
          throw new Error("Paid budget: invalid HTTP status; manual audit required")
        }
        state.http = entry.status
      } else {
        onlyKeys(entry, ["kind", "at", "attempt", "error_name", "cause_name", "cause_code"], "transport result")
        for (const key of ["error_name", "cause_name", "cause_code"]) {
          if (entry[key] !== undefined && diagnosticIdentifier(entry[key]) !== entry[key]) {
            throw new Error("Paid budget: invalid transport diagnostic; manual audit required")
          }
        }
        state.transportError = true
      }
    } else if (entry?.kind === "sse_complete") {
      const state = attempts.get(entry.attempt)
      if (!state || state.http !== 200 || state.sse !== null || state.settled !== null
        || !validUsage(entry.usage, state.reserve)
        || !/^[a-f0-9]{64}$/.test(entry.stream_sha256 ?? "")
        || chargeUnits(entry.usage.prompt_tokens, entry.usage.completion_tokens) > state.reserveUnits) {
        throw new Error("Paid budget: invalid SSE receipt; manual audit required")
      }
      state.sse = entry
    } else if (entry?.kind === "settle") {
      const state = attempts.get(entry.attempt)
      if (!state || !state.sse || state.settled
        || typeof entry.capture_id !== "string" || !entry.capture_id
        || typeof entry.observer_attempt_id !== "string" || !entry.observer_attempt_id
        || !/^[a-f0-9]{64}$/.test(entry.observer_sha256 ?? "")
        || entry.input_tokens !== state.sse.usage.prompt_tokens
        || entry.output_tokens !== state.sse.usage.completion_tokens
        || entry.charge_units !== chargeUnits(entry.input_tokens, entry.output_tokens)
        || entry.released_units !== state.reserveUnits - entry.charge_units) {
        throw new Error("Paid budget: invalid settlement; manual audit required")
      }
      state.settled = entry
      units -= entry.released_units
    } else {
      throw new Error("Paid budget: invalid ledger entry; manual audit required")
    }
  }
  if (units < 0 || units > YUAN_LIMIT * YUAN_UNITS) {
    throw new Error("Paid budget: existing ledger exceeds limit")
  }
  return { units, grossUnits, count, attempts }
}

export function createPaidBudgetGate(ledgerPath, { maxRequests = MAX_REQUESTS } = {}) {
  requirePositiveInteger(maxRequests, "maximum request count")
  if (maxRequests > MAX_REQUESTS) throw new Error("Paid budget: maximum request count exceeded")
  if (typeof ledgerPath !== "string" || !isAbsolute(ledgerPath)) {
    throw new Error("Paid budget: ledger path must be absolute")
  }
  const path = resolve(ledgerPath)
  mkdirSync(dirname(path), { recursive: true })
  const lockPath = `${path}.lock`
  let lock
  // A crash leaves this lock in place so the ledger can be checked before reuse.
  try { lock = openSync(lockPath, "wx") } catch {
    throw new Error("Paid budget: ledger is locked; manual audit required")
  }
  let ledger
  try {
    writeSync(lock, `${process.pid}\n`)
    fsyncSync(lock)
    const prior = readReservations(path)
    ledger = openSync(path, "a")
    let { units, grossUnits, count, attempts } = prior
    let closed = false
    function append(entry) {
      if (closed) throw new Error("Paid budget: ledger closed")
      const line = `${JSON.stringify(entry)}\n`
      if (writeSync(ledger, line) !== Buffer.byteLength(line)) {
        throw new Error("Paid budget: incomplete ledger write; manual audit required")
      }
      fsyncSync(ledger)
    }
    return {
      async reserve(url, input, init) {
        if (closed) throw new Error("Paid budget: ledger closed")
        const details = boundPaidRequest(url, await requestBodyText(input, init))
        if (count >= maxRequests) throw new Error("Paid budget: request count exceeded")
        const reserveUnits = chargeUnits(details.inputBound, details.outputLimit)
        if (units + reserveUnits > YUAN_LIMIT * YUAN_UNITS) {
          throw new Error("Paid budget: ¥90 reservation ceiling reached")
        }
        const attempt = count + 1
        const entry = {
          kind: "reserve", at: new Date().toISOString(), attempt,
          route: ROUTE, model: MODEL,
          request_sha256: details.requestSha256,
          request_bytes: details.requestBytes,
          input_bound: details.inputBound,
          max_completion_tokens: details.outputLimit,
          reserved_yuan: details.reservedYuan,
          cumulative_reserved_yuan: (grossUnits + reserveUnits) / YUAN_UNITS,
        }
        // Persist the reservation before a caller can send the request.
        append(entry)
        units += reserveUnits
        grossUnits += reserveUnits
        count = attempt
        attempts.set(attempt, { reserve: entry, reserveUnits, http: null,
          transportError: false, sse: null, settled: null })
        return attempt
      },
      recordHttp(attempt, status) {
        const state = attempts.get(attempt)
        if (!state || state.http !== null || state.transportError
          || !Number.isSafeInteger(status) || status < 100 || status > 599) {
          throw new Error("Paid budget: invalid HTTP result")
        }
        append({ kind: "http", at: new Date().toISOString(), attempt, status })
        state.http = status
      },
      recordTransportError(attempt, error) {
        const state = attempts.get(attempt)
        if (!state || state.http !== null || state.transportError) {
          throw new Error("Paid budget: invalid transport result")
        }
        append({ kind: "transport_error", at: new Date().toISOString(), attempt,
          ...transportDiagnostic(error) })
        state.transportError = true
      },
      recordSseComplete(attempt, usage, streamSha256) {
        const state = attempts.get(attempt)
        if (!state || state.http !== 200 || state.sse !== null
          || !validUsage(usage, state.reserve)
          || !/^[a-f0-9]{64}$/.test(streamSha256 ?? "")
          || chargeUnits(usage.prompt_tokens, usage.completion_tokens) > state.reserveUnits) {
          throw new Error("Paid budget: invalid complete SSE receipt")
        }
        const entry = { kind: "sse_complete", at: new Date().toISOString(), attempt,
          usage: { prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens }, stream_sha256: streamSha256 }
        append(entry)
        state.sse = entry
      },
      snapshot() {
        return [...attempts].map(([attempt, state]) => ({
          attempt, reserve: state.reserve, http: state.http,
          sse: state.sse, settled: state.settled,
        }))
      },
      settle(attempt, { captureId, observerAttemptId, observerSha256,
        inputTokens, outputTokens }) {
        const state = attempts.get(attempt)
        if (!state?.sse || state.settled || typeof captureId !== "string" || !captureId
          || typeof observerAttemptId !== "string" || !observerAttemptId
          || !/^[a-f0-9]{64}$/.test(observerSha256 ?? "")
          || inputTokens !== state.sse.usage.prompt_tokens
          || outputTokens !== state.sse.usage.completion_tokens) {
          throw new Error("Paid budget: settlement evidence mismatch")
        }
        const charge = chargeUnits(inputTokens, outputTokens)
        const released = state.reserveUnits - charge
        if (released < 0) throw new Error("Paid budget: settlement exceeds reservation")
        const entry = { kind: "settle", at: new Date().toISOString(), attempt,
          capture_id: captureId, observer_attempt_id: observerAttemptId,
          observer_sha256: observerSha256, input_tokens: inputTokens,
          output_tokens: outputTokens, charge_units: charge,
          released_units: released }
        append(entry)
        state.settled = entry
        units -= released
        return entry
      },
      close() {
        if (closed) return
        closed = true
        closeSync(ledger)
        closeSync(lock)
        unlinkSync(lockPath)
      },
    }
  } catch (error) {
    if (ledger !== undefined) closeSync(ledger)
    closeSync(lock)
    unlinkSync(lockPath)
    throw error
  }
}

function auditableUsage(value) {
  try {
    onlyKeys(value, ["prompt_tokens", "completion_tokens", "total_tokens",
      "prompt_tokens_details", "completion_tokens_details"], "provider usage")
    if (value.prompt_tokens_details != null) {
      onlyKeys(value.prompt_tokens_details, ["cached_tokens", "audio_tokens"], "input usage details")
      if (value.prompt_tokens_details.cached_tokens !== undefined
        && (!Number.isSafeInteger(value.prompt_tokens_details.cached_tokens)
          || value.prompt_tokens_details.cached_tokens < 0
          || value.prompt_tokens_details.cached_tokens > value.prompt_tokens)) return false
      if (value.prompt_tokens_details.audio_tokens !== undefined
        && value.prompt_tokens_details.audio_tokens !== 0) return false
    }
    if (value.completion_tokens_details != null) {
      onlyKeys(value.completion_tokens_details, ["reasoning_tokens", "audio_tokens",
        "accepted_prediction_tokens", "rejected_prediction_tokens"], "output usage details")
      for (const key of ["reasoning_tokens", "accepted_prediction_tokens", "rejected_prediction_tokens"]) {
        const tokens = value.completion_tokens_details[key]
        if (tokens !== undefined && (!Number.isSafeInteger(tokens)
          || tokens < 0 || tokens > value.completion_tokens)) return false
      }
      if (value.completion_tokens_details.audio_tokens !== undefined
        && value.completion_tokens_details.audio_tokens !== 0) return false
    }
    return Number.isSafeInteger(value.prompt_tokens) && value.prompt_tokens > 0
      && Number.isSafeInteger(value.completion_tokens) && value.completion_tokens >= 0
      && Number.isSafeInteger(value.total_tokens)
      && value.total_tokens === value.prompt_tokens + value.completion_tokens
  } catch { return false }
}

export function auditPaidSseResponse(response, gate, attempt) {
  if (response.status !== 200 || !response.body
    || !(response.headers.get("content-type") ?? "").includes("event-stream")) return response
  const decoder = new TextDecoder("utf-8", { fatal: true })
  const hash = createHash("sha256")
  let carry = ""
  let data = []
  let usage = null
  let usageCount = 0
  let doneCount = 0
  let invalid = false
  const invalidate = () => { invalid = true; carry = ""; data = [] }
  function event() {
    if (!data.length) return
    const payload = data.join("\n")
    data = []
    if (payload === "[DONE]") {
      doneCount += 1
      if (doneCount !== 1 || usageCount !== 1) invalidate()
      return
    }
    if (doneCount) { invalidate(); return }
    let frame
    try { frame = JSON.parse(payload) } catch { invalidate(); return }
    if (!frame || typeof frame !== "object" || Array.isArray(frame)) {
      invalidate(); return
    }
    if (frame.usage !== undefined && frame.usage !== null) {
      usageCount += 1
      if (usageCount !== 1 || !auditableUsage(frame.usage)) invalidate()
      else usage = frame.usage
    }
  }
  function textChunk(chunk) {
    carry += chunk
    if (carry.length > 65_536) { invalidate(); return }
    let end
    while (!invalid && (end = carry.indexOf("\n")) >= 0) {
      let line = carry.slice(0, end)
      carry = carry.slice(end + 1)
      if (line.endsWith("\r")) line = line.slice(0, -1)
      if (line.length > 65_536) { invalidate(); return }
      if (line === "") event()
      else if (line.startsWith("data:")) {
        data.push(line.slice(5).replace(/^ /, ""))
        if (data.join("\n").length > 65_536) invalidate()
      } else if (!line.startsWith(":")) invalidate()
    }
  }
  const stream = new TransformStream({
    transform(chunk, controller) {
      controller.enqueue(chunk)
      if (invalid) return
      hash.update(chunk)
      try { textChunk(decoder.decode(chunk, { stream: true })) } catch { invalidate() }
    },
    flush() {
      if (invalid) return
      try { textChunk(decoder.decode()) } catch { invalidate() }
      // Only a complete stream with one usage record can support settlement.
      if (!invalid && !carry && !data.length && doneCount === 1 && usageCount === 1) {
        gate.recordSseComplete(attempt, usage, hash.digest("hex"))
      }
    },
  })
  return new Response(response.body.pipeThrough(stream), {
    status: response.status, statusText: response.statusText, headers: response.headers,
  })
}

export function paidBudgetGateFromEnv(host, env = process.env) {
  if (!env.DSH_PAID_BUDGET_LEDGER) {
    if (isPaidPilotOrigin(host)) throw new Error("Paid budget: SUFY ledger is required")
    return null
  }
  if (host !== "https://openai.sufy.com/v1" || env.DSH_PROVIDER !== "qwen-compatible"
    || env.DSH_MODEL !== MODEL || env.DSH_MAX_RETRIES !== "0") {
    throw new Error("Paid budget: SUFY route or zero-retry configuration drift")
  }
  const maxRequests = env.DSH_PAID_BUDGET_MAX_REQUESTS === undefined
    ? MAX_REQUESTS : Number(env.DSH_PAID_BUDGET_MAX_REQUESTS)
  return createPaidBudgetGate(env.DSH_PAID_BUDGET_LEDGER, { maxRequests })
}

export function isPaidPilotOrigin(url) {
  try { return new URL(url).origin === "https://openai.sufy.com" } catch { return false }
}
