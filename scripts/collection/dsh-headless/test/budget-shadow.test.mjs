import assert from "node:assert/strict"
import { mkdtempSync, readFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { test } from "node:test"

import { createBudgetShadow } from "../budget-shadow.mjs"

function rows(path) {
  return readFileSync(path, "utf8").trim().split("\n").map(JSON.parse)
}

function setup(t, overrides = {}) {
  const dir = mkdtempSync(join(tmpdir(), "dsh-budget-shadow-"))
  t.after(() => rmSync(dir, { recursive: true, force: true }))
  const path = join(dir, "shadow.jsonl")
  const shadow = createBudgetShadow({
    path, sourceId: "exposed-dev", taskFamily: "repo_repair", runId: "r1",
    sessionId: "session-1", provider: "qwen-compatible", model: "m1",
    budgetTokens: 200, marginTokens: 64,
    measure: () => ({ totalTokens: 120, baseline: { kind: "estimated" } }),
    accountedBefore: () => 150, currentStep: () => ({ turn: 1, step: 2 }),
    now: () => "2026-09-23T00:00:00.000Z",
    ...overrides,
  })
  t.after(() => shadow.close())
  return { path, shadow }
}

test("shadow decision is durable before dispatch and does not stop a flagged call", async (t) => {
  const { path, shadow } = setup(t)
  const request = { sessionId: "session-1", provider: "qwen-compatible",
    model: "m1", messages: [{ content: "secret fixture prompt" }] }
  let dispatches = 0
  const stream = shadow.middleware(request, () => {
    dispatches += 1
    assert.equal(rows(path).at(-1).record_type, "pre_dispatch")
    assert.equal(rows(path).at(-1).would_stop, true)
    return (async function* () {
      yield { type: "usage", usage: { inputTokens: 17, outputTokens: 4,
        cacheReadTokens: 3, cacheWriteTokens: 1, reasoningTokens: 2 } }
      yield { type: "finish", reason: { kind: "completed" } }
    })()
  })
  assert.equal(dispatches, 0)
  for await (const _chunk of stream) {   }
  assert.equal(dispatches, 1)
  const records = rows(path)
  assert.equal(records[0].source_id, "exposed-dev")
  assert.equal(records[0].task_family, "repo_repair")
  assert.equal(records[1].projected_provider_accounted_tokens, 206)
  assert.equal(records[1].policy_action, "pass_through")
  assert.deepEqual(records[2].usage, {
    input_tokens: 17, output_tokens: 4, cache_read_tokens: 3,
    cache_write_tokens: 1, reasoning_output_tokens: 2,
    provider_accounted_tokens: 25,
  })
  assert.equal(records[2].yuan_spend, null)
  assert.equal(readFileSync(path, "utf8").includes("secret fixture prompt"), false)
})

test("retry is separately traced and an unrelated session is untouched", async (t) => {
  const { path, shadow } = setup(t)
  const request = { sessionId: "session-1", provider: "qwen-compatible", model: "m1" }
  let count = 0
  const next = () => {
    count += 1
    return (async function* () { yield { type: "finish", reason: { kind: "failed" } } })()
  }
  for await (const _chunk of shadow.middleware(request, next)) {   }
  for await (const _chunk of shadow.middleware(request, next)) {   }
  for await (const _chunk of shadow.middleware({ ...request, sessionId: "other" }, next)) {   }
  assert.equal(count, 3)
  const pre = rows(path).filter((row) => row.record_type === "pre_dispatch")
  assert.deepEqual(pre.map((row) => [row.attempt_ordinal, row.retry]), [[1, false], [2, true]])
  assert.equal(rows(path).filter((row) => row.record_type === "attempt_end" && row.usage_status === "missing").length, 2)
})

test("missing meter evidence and route drift stay visible without enforcing a stop", async (t) => {
  const { path, shadow } = setup(t, { measure: () => ({ baseline: { kind: "none" } }) })
  let dispatches = 0
  const request = { sessionId: "session-1", provider: "qwen-compatible", model: "m1" }
  const next = () => {
    dispatches += 1
    return (async function* () { yield { type: "finish", reason: { kind: "completed" } } })()
  }
  for await (const _chunk of shadow.middleware(request, next)) {   }
  for await (const _chunk of shadow.middleware({ ...request, model: "other" }, next)) {   }
  assert.equal(dispatches, 2)
  const records = rows(path)
  assert.equal(records[1].decision_status, "unavailable")
  assert.equal(records[1].would_stop, null)
  assert.equal(records[1].policy_action, "pass_through")
  assert.equal(records.at(-1).record_type, "route_mismatch")
  assert.equal(records.at(-1).policy_action, "pass_through")
})
