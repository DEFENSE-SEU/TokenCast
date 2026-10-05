import { importDshHostModule } from "./host.mjs"
import { Agent } from "undici"
import { auditPaidSseResponse, isPaidPilotOrigin, paidBudgetGateFromEnv } from "./paid-budget-gate.mjs"

const schemastery = await importDshHostModule("@deepseek-ai/schemastery")
const z = schemastery.default ?? schemastery

const GEMINI_SIGNATURE_FORMAT = "token-prediction.gemini-thought-signature.v1"

function encryptedDetail(detail) {
  return detail?.type === "reasoning.encrypted"
    && typeof detail.id === "string" && detail.id.length > 0
    && typeof detail.data === "string" && detail.data.length > 0
}

function bridgeGeminiSignatures(delta, choiceState) {
  const affected = new Set()
  const details = Array.isArray(delta.reasoning_details) ? delta.reasoning_details : []
  for (const detail of details) {
    if (!encryptedDetail(detail)) continue
    const originals = choiceState.details.get(detail.id) ?? []
    if (!originals.some((item) => JSON.stringify(item) === JSON.stringify(detail))) {
      originals.push(detail)
    }
    choiceState.details.set(detail.id, originals)
    for (const call of choiceState.calls.values()) {
      if (call.id === detail.id && call.signature) affected.add(call)
    }
  }
  const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : []
  for (const [position, call] of calls.entries()) {
    if (!call || typeof call !== "object") continue
    const index = call.index ?? position
    const known = choiceState.calls.get(index) ?? {}
    const previousId = known.id
    const previousSignature = known.signature
    if (typeof call.id === "string" && call.id) known.id = call.id
    const signature = call.extra_content?.google?.thought_signature
    if (typeof signature === "string" && signature) known.signature = signature
    choiceState.calls.set(index, known)
    if (known.id && known.signature
      && (known.id !== previousId || known.signature !== previousSignature)) affected.add(known)
  }
  if (affected.size === 0) return delta
  return {
    ...delta,
    reasoning_details: [
      ...details,
      ...[...affected].map((call) => ({
        type: "reasoning.encrypted",
        id: call.id,
        data: call.signature,
        format: GEMINI_SIGNATURE_FORMAT,
        original_reasoning_details: choiceState.details.get(call.id) ?? [],
      })),
    ],
  }
}

export function repairRequestBody(body) {
  if (!Array.isArray(body?.messages)) return body
  let changed = false
  const messages = body.messages.map((message) => {
    if (message?.role !== "assistant" || !Array.isArray(message.reasoning_details)) {
      return message
    }
    const carriers = message.reasoning_details.filter(
      (detail) => detail?.format === GEMINI_SIGNATURE_FORMAT,
    )
    if (carriers.length === 0) return message
    const calls = [...(message.tool_calls ?? [])]
    for (const carrier of carriers) {
      if (!encryptedDetail(carrier) || !Array.isArray(carrier.original_reasoning_details)) {
        throw new Error("Invalid Gemini signature replay metadata")
      }
      const index = calls.findIndex((call) => call.id === carrier.id)
      if (index < 0) throw new Error("Gemini signature replay has no matching tool call")
      const call = calls[index]
      const existing = call.extra_content?.google?.thought_signature
      if (existing !== undefined && existing !== carrier.data) {
        throw new Error("Conflicting Gemini tool-call signatures in replay")
      }
      calls[index] = {
        ...call,
        extra_content: {
          ...call.extra_content,
          google: { ...call.extra_content?.google, thought_signature: carrier.data },
        },
      }
    }
    changed = true
    const repaired = { ...message, tool_calls: calls }
    const remaining = message.reasoning_details.flatMap((detail) =>
      detail?.format === GEMINI_SIGNATURE_FORMAT ? detail.original_reasoning_details : [detail],
    )
    if (remaining.length > 0) repaired.reasoning_details = remaining
    else delete repaired.reasoning_details
    return repaired
  })
  return changed ? { ...body, messages } : body
}

function isBlank(value) {
  return value === "" || value === null
}

function repairToolCall(call) {
  if (!call || typeof call !== "object") return call
  const repaired = { ...call }
  if (isBlank(repaired.id)) delete repaired.id
  if (repaired.function && typeof repaired.function === "object") {
    const fn = { ...repaired.function }
    if (isBlank(fn.name)) delete fn.name
    repaired.function = fn
  }
  return repaired
}

export function repairChunk(chunk) {
  return repairChunkWithState(chunk, new Map())
}

function repairChunkWithState(chunk, signatureState) {
  if (!chunk || typeof chunk !== "object") return chunk
  const choices = chunk.choices
  if (!Array.isArray(choices)) return chunk
  let touched = false
  const next = choices.map((choice, position) => {
    const delta = choice?.delta
    if (!delta || typeof delta !== "object") return choice
    const key = choice.index ?? position
    const state = signatureState.get(key) ?? { calls: new Map(), details: new Map() }
    signatureState.set(key, state)
    const bridged = bridgeGeminiSignatures(delta, state)
    const calls = bridged.tool_calls
    if (!Array.isArray(calls) || calls.length === 0) {
      if (bridged === delta) return choice
      touched = true
      return { ...choice, delta: bridged }
    }
    touched = true
    return { ...choice, delta: { ...bridged, tool_calls: calls.map(repairToolCall) } }
  })
  return touched ? { ...chunk, choices: next } : chunk
}

export function repairSseLine(line) {
  return repairSseLineWithState(line, new Map())
}

function repairSseLineWithState(line, signatureState) {
  if (!line.startsWith("data:")) return line
  const payload = line.slice(5).trim()
  if (!payload || payload === "[DONE]") return line
  let parsed
  try {
    parsed = JSON.parse(payload)
  } catch {
    return line
  }
  const repaired = repairChunkWithState(parsed, signatureState)
  return repaired === parsed ? line : `data: ${JSON.stringify(repaired)}`
}

export function repairResponse(response) {
  if (!response.body) return response
  const decoder = new TextDecoder()
  const encoder = new TextEncoder()
  const signatureState = new Map()
  let carry = ""
  const repairLine = (line) =>
    line.endsWith("\r")
      ? `${repairSseLineWithState(line.slice(0, -1), signatureState)}\r`
      : repairSseLineWithState(line, signatureState)
  const stream = new TransformStream({
    transform(chunk, controller) {
      carry += decoder.decode(chunk, { stream: true })
      const boundary = carry.lastIndexOf("\n")
      if (boundary < 0) return
      const complete = carry.slice(0, boundary + 1)
      carry = carry.slice(boundary + 1)
      controller.enqueue(
        encoder.encode(complete.split("\n").map(repairLine).join("\n")),
      )
    },
    flush(controller) {
      if (carry) controller.enqueue(encoder.encode(repairLine(carry)))
    },
  })
  return new Response(response.body.pipeThrough(stream), {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  })
}

function requestUrl(input) {
  if (typeof input === "string") return input
  if (input instanceof URL) return input.href
  return input?.url ?? ""
}

async function repairFetchRequest(input, init) {
  const isRequest = input instanceof Request
  const body = init?.body ?? (isRequest ? await input.clone().text() : undefined)
  if (typeof body !== "string") return [input, init]
  let parsed
  try {
    parsed = JSON.parse(body)
  } catch {
    return [input, init]
  }
  const repaired = repairRequestBody(parsed)
  if (repaired === parsed) return [input, init]
  const headers = new Headers(init?.headers ?? (isRequest ? input.headers : undefined))
  headers.delete("content-length")
  if (isRequest && init?.body === undefined) {
    return [
      new Request(input, { body: JSON.stringify(repaired), headers }),
      init?.headers === undefined ? init : { ...init, headers },
    ]
  }
  return [input, { ...init, body: JSON.stringify(repaired), headers }]
}

export const name = "aggregator-adapter"
export const Config = z.object({
  host: z.string().default(""),
})

export function apply(ctx, config) {
  ctx.effect(() => {
    const host = config.host || process.env.DSH_AGGREGATOR_HOST || ""
    const budget = paidBudgetGateFromEnv(host)
    const paidDispatcher = budget ? new Agent({ connect: { timeout: 30_000 } }) : null
    const original = globalThis.fetch
    globalThis.fetch = async function repairedFetch(input, init) {
      const url = requestUrl(input)
      if (budget && url !== "https://openai.sufy.com/v1/chat/completions"
        && (isPaidPilotOrigin(url) || url.includes("/chat/completions"))) {
        throw new Error("Paid budget: unexpected SUFY endpoint")
      }
      if ((host && !url.includes(host)) || !url.includes("/chat/completions")) {
        return original.call(this, input, init)
      }
      const [request, options] = await repairFetchRequest(input, init)
      if (budget && options?.body !== undefined && typeof options.body !== "string") {
        throw new Error("Paid budget: non-string body override is not auditable")
      }
      const finalRequest = budget ? new Request(request, { ...options, redirect: "error" }) : null
      if (finalRequest && finalRequest.method !== "POST") {
        throw new Error("Paid budget: provider method must be POST")
      }
      const attempt = budget ? await budget.reserve(finalRequest.url, finalRequest) : null
      let response
      try {
        response = await original.call(this, finalRequest ?? request,
          budget ? { dispatcher: paidDispatcher } : options)
      } catch (error) {
        if (attempt !== null) budget.recordTransportError(attempt, error)
        throw error
      }
      if (attempt !== null) budget.recordHttp(attempt, response.status)
      if (!(response.headers.get("content-type") ?? "").includes("event-stream")) {
        return response
      }
      return repairResponse(budget ? auditPaidSseResponse(response, budget, attempt) : response)
    }
    return () => {
      globalThis.fetch = original
      budget?.close()
      if (paidDispatcher) void paidDispatcher.close()
    }
  }, "aggregator-adapter")
}
