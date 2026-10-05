import assert from "node:assert/strict"
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import test from "node:test"
import { Agent } from "undici"

import { apply } from "../aggregator-adapter.mjs"
import { boundPaidRequest, createPaidBudgetGate, paidBudgetGateFromEnv } from "../paid-budget-gate.mjs"

const ROUTE = "https://openai.sufy.com/v1/chat/completions"
function body(overrides = {}) {
  return JSON.stringify({
    model: "openai/gpt-5.4", stream: true, max_completion_tokens: 512,
    reasoning_effort: "low", stream_options: { include_usage: true }, store: false,
    messages: [{ role: "user", content: "offline fixture" }],
    ...overrides,
  })
}

test("bounds the final wire body and refuses absent or ambiguous output caps", () => {
  const valid = body()
  const bound = boundPaidRequest(ROUTE, valid)
  assert.equal(bound.requestBytes, Buffer.byteLength(valid))
  assert.equal(bound.inputBound, Buffer.byteLength(valid) + 4096)
  assert.equal(bound.outputLimit, 512)
  assert.ok(bound.reservedYuan > 0)
  assert.throws(() => boundPaidRequest(ROUTE, body({ max_completion_tokens: undefined })), /invalid max_completion_tokens/)
  assert.throws(() => boundPaidRequest(ROUTE, body({ max_tokens: 512 })), /ambiguous output limit/)
  assert.throws(() => boundPaidRequest(ROUTE, body({ temperature: 0 })), /unsupported temperature/)
  assert.throws(() => boundPaidRequest(ROUTE, body({ max_completion_tokens: 1281 })), /output limit exceeded/)
  assert.throws(() => boundPaidRequest(ROUTE, body({ messages: [{ role: "user", content: [
    { type: "image_url", image_url: { url: "https://example.test/image" } },
  ] }] })), /non-text message/)
  assert.ok(boundPaidRequest(ROUTE, body({ messages: [{ role: "user", content: "x".repeat(35_000) }] })).inputBound <= 40_000)
  assert.throws(() => boundPaidRequest(ROUTE, body({ messages: [{ role: "user", content: "x".repeat(37_000) }] })), /input bound exceeded/)
  assert.throws(() => boundPaidRequest("https://other.example/v1/chat/completions", valid), /unexpected provider route/)
  assert.throws(() => boundPaidRequest(ROUTE, body({ reasoning_effort: "high" })), /reasoning effort/)
  assert.throws(() => boundPaidRequest(ROUTE, body({ modalities: ["text", "audio"], audio: { voice: "alloy" } })), /unsupported request field/)
  assert.throws(() => boundPaidRequest(ROUTE, body({ messages: [
    { role: "user", content: "hello", image_url: "https://example.test/image" },
  ] })), /unsupported message field/)
  assert.throws(() => boundPaidRequest(ROUTE, body({ messages: [
    { role: "user", content: [{ type: "text", text: "hello", image_url: "x" }] },
  ] })), /non-text message content/)
})

test("first and tool-history requests retain the bounded text-only wire shape", () => {
  const tool = { type: "function", function: {
    name: "pwsh", description: "Run a shell command", parameters: { type: "object" }, strict: false,
  } }
  const first = body({ tools: [tool], messages: [
    { role: "system", content: "system" }, { role: "user", content: "task" },
  ] })
  const second = body({ tools: [tool], messages: [
    { role: "system", content: "system" }, { role: "user", content: "task" },
    { role: "assistant", content: null, tool_calls: [{
      id: "call_1", type: "function", function: { name: "pwsh", arguments: "{}" },
    }] },
    { role: "tool", tool_call_id: "call_1", content: "ok" },
  ] })
  assert.ok(boundPaidRequest(ROUTE, first).reservedYuan > 0)
  assert.ok(boundPaidRequest(ROUTE, second).reservedYuan > 0)
})

test("SUFY route requires a persistent ledger even when no ledger variable is set", () => {
  const env = { DSH_PROVIDER: "qwen-compatible", DSH_MODEL: "openai/gpt-5.4", DSH_MAX_RETRIES: "0" }
  assert.throws(() => paidBudgetGateFromEnv("https://openai.sufy.com/v1", env), /SUFY ledger is required/)
  assert.equal(paidBudgetGateFromEnv("https://other.example/v1", env), null)
})

test("reservations persist across runs and a second fetch is refused before network", async (t) => {
  const dir = mkdtempSync(join(tmpdir(), "paid-gate-"))
  t.after(() => rmSync(dir, { recursive: true, force: true }))
  const ledgerPath = join(dir, "reservations.jsonl")
  const envNames = ["DSH_PAID_BUDGET_LEDGER", "DSH_PAID_BUDGET_MAX_REQUESTS", "DSH_PROVIDER", "DSH_MODEL", "DSH_MAX_RETRIES"]
  const priorEnv = Object.fromEntries(envNames.map((name) => [name, process.env[name]]))
  Object.assign(process.env, {
    DSH_PAID_BUDGET_LEDGER: ledgerPath,
    DSH_PAID_BUDGET_MAX_REQUESTS: "1",
    DSH_PROVIDER: "qwen-compatible",
    DSH_MODEL: "openai/gpt-5.4",
    DSH_MAX_RETRIES: "0",
  })
  t.after(() => {
    for (const name of envNames) {
      if (priorEnv[name] === undefined) delete process.env[name]
      else process.env[name] = priorEnv[name]
    }
  })
  const sends = []
  t.mock.method(globalThis, "fetch", async (input, init) => {
    sends.push({ input, init })
    return new Response("{}", { status: 200, headers: { "content-type": "application/json" } })
  })
  let dispose
  apply({ effect: (start) => { dispose = start() } }, { host: "https://openai.sufy.com/v1" })
  try {
    const request = new Request(ROUTE, { method: "POST", body: body(), headers: { "content-type": "application/json" } })
    await fetch(request)
    assert.equal(sends.length, 1)
    assert.equal(sends[0].input.redirect, "error")
    assert.equal(await sends[0].input.clone().text(), body())
    assert.ok(sends[0].init.dispatcher instanceof Agent)
    assert.deepEqual(Object.keys(sends[0].init), ["dispatcher"])
    assert.ok(Reflect.ownKeys(sends[0].init.dispatcher).some((key) =>
      sends[0].init.dispatcher[key]?.connect?.timeout === 30_000))
    const otherInit = { method: "HEAD" }
    await fetch("https://unrelated.example.test/health", otherInit)
    assert.strictEqual(sends[1].init, otherInit)
    await assert.rejects(fetch(new Request(ROUTE, { method: "POST", body: body() })),
      /request count exceeded/)
    assert.equal(sends.length, 2)
  } finally {
    dispose()
  }
  assert.equal(existsSync(`${ledgerPath}.lock`), false)
})

test("complete provider usage is durable before the next pre-send rejection", async (t) => {
  const dir = mkdtempSync(join(tmpdir(), "paid-gate-"))
  t.after(() => rmSync(dir, { recursive: true, force: true }))
  const ledgerPath = join(dir, "reservations.jsonl")
  const envNames = ["DSH_PAID_BUDGET_LEDGER", "DSH_PAID_BUDGET_MAX_REQUESTS", "DSH_PROVIDER", "DSH_MODEL", "DSH_MAX_RETRIES"]
  const priorEnv = Object.fromEntries(envNames.map((name) => [name, process.env[name]]))
  Object.assign(process.env, {
    DSH_PAID_BUDGET_LEDGER: ledgerPath,
    DSH_PAID_BUDGET_MAX_REQUESTS: "1",
    DSH_PROVIDER: "qwen-compatible",
    DSH_MODEL: "openai/gpt-5.4",
    DSH_MAX_RETRIES: "0",
  })
  t.after(() => {
    for (const name of envNames) {
      if (priorEnv[name] === undefined) delete process.env[name]
      else process.env[name] = priorEnv[name]
    }
  })
  const usage = { prompt_tokens: 100, completion_tokens: 7, total_tokens: 107 }
  const stream = `data: ${JSON.stringify({ id: "fixture", choices: [], usage })}\n\ndata: [DONE]\n\n`
  let sends = 0
  t.mock.method(globalThis, "fetch", async () => {
    sends += 1
    return new Response(stream, { status: 200, headers: { "content-type": "text/event-stream" } })
  })
  let dispose
  apply({ effect: (start) => { dispose = start() } }, { host: "https://openai.sufy.com/v1" })
  try {
    await (await fetch(new Request(ROUTE, { method: "POST", body: body() }))).text()
    const beforeSecond = readFileSync(ledgerPath, "utf8").trim().split("\n").map(JSON.parse)
    assert.deepEqual(beforeSecond.map((entry) => entry.kind), ["reserve", "http", "sse_complete"])
    assert.deepEqual(beforeSecond[2].usage, usage)
    await assert.rejects(fetch(new Request(ROUTE, { method: "POST", body: body() })),
      /request count exceeded/)
    assert.equal(sends, 1)
    assert.deepEqual(readFileSync(ledgerPath, "utf8").trim().split("\n").map(JSON.parse), beforeSecond)
  } finally {
    dispose()
  }
  const restored = createPaidBudgetGate(ledgerPath)
  try { assert.deepEqual(restored.snapshot()[0].sse.usage, usage) } finally { restored.close() }
})

test("a body override cannot change the sent bytes and paid fetch forbids redirects", async (t) => {
  const dir = mkdtempSync(join(tmpdir(), "paid-gate-"))
  t.after(() => rmSync(dir, { recursive: true, force: true }))
  const ledgerPath = join(dir, "reservations.jsonl")
  const envNames = ["DSH_PAID_BUDGET_LEDGER", "DSH_PROVIDER", "DSH_MODEL", "DSH_MAX_RETRIES"]
  const priorEnv = Object.fromEntries(envNames.map((name) => [name, process.env[name]]))
  Object.assign(process.env, {
    DSH_PAID_BUDGET_LEDGER: ledgerPath,
    DSH_PROVIDER: "qwen-compatible",
    DSH_MODEL: "openai/gpt-5.4",
    DSH_MAX_RETRIES: "0",
  })
  t.after(() => {
    for (const name of envNames) {
      if (priorEnv[name] === undefined) delete process.env[name]
      else process.env[name] = priorEnv[name]
    }
  })
  let sends = 0
  t.mock.method(globalThis, "fetch", async (request) => {
    sends += 1
    assert.equal(request.redirect, "error")
    assert.equal(await request.clone().text(), body())
    return new Response(null, { status: 307, headers: { location: ROUTE } })
  })
  let dispose
  apply({ effect: (start) => { dispose = start() } }, { host: "https://openai.sufy.com/v1" })
  try {
    const request = new Request(ROUTE, { method: "POST", body: body() })
    await assert.rejects(fetch(request, { body: new TextEncoder().encode("x".repeat(30_000)) }),
      /non-string body override/)
    assert.equal(sends, 0)
    assert.equal(readFileSync(ledgerPath, "utf8"), "")
    const response = await fetch(request.clone())
    assert.equal(response.status, 307)
    assert.equal(sends, 1)
  } finally {
    dispose()
  }
  const entries = readFileSync(ledgerPath, "utf8").trim().split("\n").map(JSON.parse)
  assert.deepEqual(entries.map((entry) => entry.kind), ["reserve", "http"])
})

test("transport diagnostics record only safe error identifiers", async (t) => {
  const dir = mkdtempSync(join(tmpdir(), "paid-gate-"))
  t.after(() => rmSync(dir, { recursive: true, force: true }))
  const ledgerPath = join(dir, "reservations.jsonl")
  const envNames = ["DSH_PAID_BUDGET_LEDGER", "DSH_PROVIDER", "DSH_MODEL", "DSH_MAX_RETRIES"]
  const priorEnv = Object.fromEntries(envNames.map((name) => [name, process.env[name]]))
  Object.assign(process.env, {
    DSH_PAID_BUDGET_LEDGER: ledgerPath,
    DSH_PROVIDER: "qwen-compatible",
    DSH_MODEL: "openai/gpt-5.4",
    DSH_MAX_RETRIES: "0",
  })
  t.after(() => {
    for (const name of envNames) {
      if (priorEnv[name] === undefined) delete process.env[name]
      else process.env[name] = priorEnv[name]
    }
  })
  const timeout = new TypeError("secret-token-and-url")
  timeout.cause = Object.assign(new Error("secret-header"), {
    name: "ConnectTimeoutError", code: "UND_ERR_CONNECT_TIMEOUT",
  })
  const reset = new TypeError("secret-key")
  reset.cause = Object.assign(new Error("secret-address"), { code: "ECONNRESET" })
  const unsafe = new Error("secret-message")
  unsafe.name = "TypeError\nAuthorization: secret-key"
  unsafe.cause = { name: "ConnectTimeoutError\nsecret-header", code: "ECONNRESET;secret-token" }
  const failures = [timeout, reset, unsafe]
  t.mock.method(globalThis, "fetch", async () => { throw failures.shift() })
  let dispose
  apply({ effect: (start) => { dispose = start() } }, { host: "https://openai.sufy.com/v1" })
  try {
    for (const error of [timeout, reset, unsafe]) {
      await assert.rejects(fetch(new Request(ROUTE, { method: "POST", body: body() })),
        (caught) => caught === error)
    }
  } finally {
    dispose()
  }
  const text = readFileSync(ledgerPath, "utf8")
  for (const secret of ["secret-token", "secret-header", "secret-key", "secret-address", "secret-message", "Authorization"]) {
    assert.equal(text.includes(secret), false)
  }
  const entries = text.trim().split("\n").map(JSON.parse)
  assert.deepEqual(entries.map((entry) => entry.kind), [
    "reserve", "transport_error", "reserve", "transport_error", "reserve", "transport_error",
  ])
  assert.deepEqual(entries[1], {
    kind: "transport_error", at: entries[1].at, attempt: 1,
    error_name: "TypeError", cause_name: "ConnectTimeoutError", cause_code: "UND_ERR_CONNECT_TIMEOUT",
  })
  assert.deepEqual(entries[3], {
    kind: "transport_error", at: entries[3].at, attempt: 2,
    error_name: "TypeError", cause_name: "Error", cause_code: "ECONNRESET",
  })
  assert.deepEqual(entries[5], { kind: "transport_error", at: entries[5].at, attempt: 3 })
  entries[5] = { kind: "transport_error", at: entries[5].at, attempt: 3 }
  writeFileSync(ledgerPath, `${entries.map(JSON.stringify).join("\n")}\n`)
  const reopened = createPaidBudgetGate(ledgerPath)
  try { assert.equal(reopened.snapshot().length, 3) } finally { reopened.close() }
})

test("ledger refuses reuse above the persisted request count", async (t) => {
  const dir = mkdtempSync(join(tmpdir(), "paid-gate-"))
  t.after(() => rmSync(dir, { recursive: true, force: true }))
  const path = join(dir, "reservations.jsonl")
  const first = createPaidBudgetGate(path, { maxRequests: 1 })
  const attempt = await first.reserve(ROUTE, new Request(ROUTE, { method: "POST", body: body() }))
  first.recordHttp(attempt, 200)
  first.close()
  const second = createPaidBudgetGate(path, { maxRequests: 1 })
  try {
    await assert.rejects(second.reserve(ROUTE, new Request(ROUTE, { method: "POST", body: body() })), /request count exceeded/)
  } finally {
    second.close()
  }
  const entries = readFileSync(path, "utf8").trim().split("\n").map(JSON.parse)
  assert.deepEqual(entries.map((entry) => entry.kind), ["reserve", "http"])
  assert.equal(entries[0].max_completion_tokens, 512)
  assert.equal(entries[0].attempt, 1)
  assert.ok(entries[0].cumulative_reserved_yuan < 90)
})

test("a restored ledger stops before the ¥90 reservation ceiling", async (t) => {
  const dir = mkdtempSync(join(tmpdir(), "paid-gate-"))
  t.after(() => rmSync(dir, { recursive: true, force: true }))
  const path = join(dir, "reservations.jsonl")
  const reserved = 5 * 8 * (2.5 * 20_000 + 15 * 1280) / 1_000_000
  writeFileSync(path, Array.from({ length: 32 }, (_, index) => JSON.stringify({
    kind: "reserve", attempt: index + 1,
    route: ROUTE, model: "openai/gpt-5.4",
    input_bound: 20_000, max_completion_tokens: 1280,
    reserved_yuan: reserved,
  })).join("\n") + "\n")
  const gate = createPaidBudgetGate(path)
  try {
    const large = body({ max_completion_tokens: 1280,
      messages: [{ role: "user", content: "x".repeat(10_000) }] })
    await assert.rejects(gate.reserve(ROUTE, new Request(ROUTE, { method: "POST", body: large })),
      /¥90 reservation ceiling reached/)
  } finally {
    gate.close()
  }
  assert.equal(readFileSync(path, "utf8").trim().split("\n").length, 32)
})
