import assert from "node:assert/strict"
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import test from "node:test"

import { auditPaidSseResponse, createPaidBudgetGate } from "../paid-budget-gate.mjs"
import { settlePaidBudgetCapture } from "../settle-paid-budget.mjs"

const ROUTE = "https://openai.sufy.com/v1/chat/completions"
const REQUEST = JSON.stringify({ model: "openai/gpt-5.4", stream: true,
  stream_options: { include_usage: true }, reasoning_effort: "low", store: false,
  max_completion_tokens: 512, messages: [{ role: "user", content: "fixture" }] })
const USAGE = { prompt_tokens: 100, completion_tokens: 7, total_tokens: 107 }
const frame = (value) => `data: ${JSON.stringify(value)}\n\n`
const usageFrame = frame({ id: "chatcmpl-fixture", choices: [], usage: USAGE })
const doneFrame = "data: [DONE]\n\n"

function setup(t) {
  const dir = mkdtempSync(join(tmpdir(), "paid-settle-"))
  t.after(() => rmSync(dir, { recursive: true, force: true }))
  return { dir, ledgerPath: join(dir, "ledger.jsonl") }
}

async function reserved(gate) {
  return gate.reserve(ROUTE, new Request(ROUTE, { method: "POST", body: REQUEST }))
}

function entries(path) {
  return readFileSync(path, "utf8").trim().split("\n").map(JSON.parse)
}

function writeCapture(dir, start, end, usage = USAGE) {
  const capturePath = join(dir, "capture.json")
  const sessionId = "session-fixture"
  writeFileSync(capturePath, JSON.stringify({
    capture_id: "capture-fixture", session_id: sessionId,
    condition: { configured_model: "openai/gpt-5.4", configured_provider: "qwen-compatible",
      reasoning_effort: "low" },
    runtime: { hosted_provider: { base_url: "https://openai.sufy.com/v1" },
      max_attempts_per_logical_call: 1, opaque_provider_retry_allowed: false },
    terminal: { attempt_ledger_complete: true },
  }))
  const rows = [
    { record_type: "observer/session", session_id: sessionId },
    { record_type: "attempt/start", session_id: sessionId,
      attempt_id: "attempt-fixture", logical_call_id: "call-fixture", attempt_ordinal: 1,
      provider: "qwen-compatible", model: "openai/gpt-5.4", reasoning_effort: "low", time: start },
    { record_type: "attempt/end", session_id: sessionId,
      attempt_id: "attempt-fixture", logical_call_id: "call-fixture", time: end,
      failure_code: null, observer_finalized: true,
      usage: { inputTokens: usage.prompt_tokens, outputTokens: usage.completion_tokens,
        cacheReadTokens: null, cacheWriteTokens: null } },
  ]
  writeFileSync(join(dir, "observer.jsonl"), `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`)
  return capturePath
}

test("raw SSE audit streams chunks unchanged and records only complete usage at EOF", async (t) => {
  const { ledgerPath } = setup(t)
  const gate = createPaidBudgetGate(ledgerPath)
  try {
    const attempt = await reserved(gate)
    gate.recordHttp(attempt, 200)
    let source
    const input = new ReadableStream({ start(controller) { source = controller } })
    const audited = auditPaidSseResponse(new Response(input, {
      status: 200, headers: { "content-type": "text/event-stream" },
    }), gate, attempt)
    const reader = audited.body.getReader()
    const first = new TextEncoder().encode(frame({ choices: [{ delta: { content: "a" } }] }))
    source.enqueue(first)
    const delivered = await reader.read()
    assert.equal(new TextDecoder().decode(delivered.value), new TextDecoder().decode(first))
    assert.equal(entries(ledgerPath).length, 2)
    source.enqueue(new TextEncoder().encode(usageFrame.slice(0, 24)))
    source.enqueue(new TextEncoder().encode(usageFrame.slice(24) + doneFrame))
    source.close()
    while (!(await reader.read()).done) {}
    const receipt = entries(ledgerPath).at(-1)
    assert.equal(receipt.kind, "sse_complete")
    assert.equal(receipt.attempt, attempt)
    assert.deepEqual(receipt.usage, USAGE)
    assert.match(receipt.stream_sha256, /^[a-f0-9]{64}$/)
  } finally { gate.close() }
})

test("missing, duplicated, and interrupted SSE usage retain the full reservation", async (t) => {
  const { ledgerPath } = setup(t)
  const gate = createPaidBudgetGate(ledgerPath)
  try {
    for (const raw of [usageFrame, usageFrame + usageFrame + doneFrame,
      frame({ choices: [{ delta: { content: "a" } }] }) + doneFrame]) {
      const attempt = await reserved(gate)
      gate.recordHttp(attempt, 200)
      const audited = auditPaidSseResponse(new Response(raw, {
        status: 200, headers: { "content-type": "text/event-stream" },
      }), gate, attempt)
      assert.equal(await audited.text(), raw)
    }
    const attempt = await reserved(gate)
    gate.recordHttp(attempt, 200)
    const broken = new ReadableStream({ start(controller) {
      controller.enqueue(new TextEncoder().encode(usageFrame))
      controller.error(new Error("stream lost"))
    } })
    await assert.rejects(auditPaidSseResponse(new Response(broken, {
      status: 200, headers: { "content-type": "text/event-stream" },
    }), gate, attempt).text(), /stream lost/)
    assert.equal(entries(ledgerPath).filter((entry) => entry.kind === "sse_complete").length, 0)
  } finally { gate.close() }
})

test("capture reconciliation settles one matched attempt once and survives restart", async (t) => {
  const { dir, ledgerPath } = setup(t)
  const gate = createPaidBudgetGate(ledgerPath)
  const start = Date.now() - 1000
  let reservedYuan
  try {
    const attempt = await reserved(gate)
    reservedYuan = entries(ledgerPath)[0].reserved_yuan
    gate.recordHttp(attempt, 200)
    const audited = auditPaidSseResponse(new Response(usageFrame + doneFrame, {
      status: 200, headers: { "content-type": "text/event-stream" },
    }), gate, attempt)
    await audited.text()
  } finally { gate.close() }
  const capturePath = writeCapture(dir, start, Date.now() + 1000)
  const result = settlePaidBudgetCapture(ledgerPath, capturePath)
  assert.equal(result.settled, 1)
  assert.equal(Math.round(result.releasedYuan * 10_000),
    Math.round(reservedYuan * 10_000) - (100 + 6 * 7))
  assert.equal(settlePaidBudgetCapture(ledgerPath, capturePath).settled, 0)
  assert.equal(entries(ledgerPath).filter((entry) => entry.kind === "settle").length, 1)
  const restored = createPaidBudgetGate(ledgerPath)
  restored.close()
})

test("unreceipted old calls and capture usage mismatch cannot release money", async (t) => {
  const { dir, ledgerPath } = setup(t)
  const gate = createPaidBudgetGate(ledgerPath)
  const start = Date.now() - 1000
  try {
    const attempt = await reserved(gate)
    gate.recordHttp(attempt, 200)
  } finally { gate.close() }
  const capturePath = writeCapture(dir, start, Date.now() + 1000)
  assert.equal(settlePaidBudgetCapture(ledgerPath, capturePath).settled, 0)
  assert.equal(entries(ledgerPath).filter((entry) => entry.kind === "settle").length, 0)
  const fresh = createPaidBudgetGate(ledgerPath)
  try { fresh.recordSseComplete(1, USAGE, "a".repeat(64)) } finally { fresh.close() }
  writeCapture(dir, start, Date.now() + 1000,
    { prompt_tokens: USAGE.prompt_tokens + 1, completion_tokens: 7 })
  assert.throws(() => settlePaidBudgetCapture(ledgerPath, capturePath), /usage disagree/)
  assert.equal(entries(ledgerPath).filter((entry) => entry.kind === "settle").length, 0)
})
