import assert from "node:assert/strict"
import { mkdtemp, readFile, rm } from "node:fs/promises"
import { createRequire } from "node:module"
import { homedir, tmpdir } from "node:os"
import { join } from "node:path"
import { pathToFileURL } from "node:url"
import test from "node:test"

import {
  canonicalRequestJson, createToolResultContentPolicy, disposeRequestContentSources,
  RequestContentLedger, retainToolResultContent,
  pruneToolResultHistory, registerRequestHistoryPolicy,
} from "../request-content-ledger.mjs"

const text = (value) => ({ type: "text", text: value })
const request = (messages) => ({ provider: "fixture", model: "fixture", sessionId: "s", messages, tools: [], system: "system" })
const message = (id, value) => ({ id, role: "user", content: [text(value)] })
const facts = (attempt = 1, step = 1) => ({ sessionId: "s", logicalCallId: `s:turn-1:step-${step}`, attemptId: `s:turn-1:step-${step}:attempt-${attempt}`, attemptOrdinal: attempt })
const makeLedger = () => {
  const records = []
  const ledger = new RequestContentLedger({ writeRecord: (_id, record) => records.push(record) })
  ledger.observeNativeEvent("s", { seq: 8 })
  return { ledger, records }
}

test("retention is a UTF-8 scalar prefix across blocks and preserves the original", () => {
  const original = Object.freeze([Object.freeze(text("a😀\u4e2d")), Object.freeze(text("tail"))])
  assert.deepEqual(retainToolResultContent(original, 5), [text("a😀"), text("")])
  assert.deepEqual(retainToolResultContent(original, 4), [text("a"), text("")])
  assert.deepEqual(retainToolResultContent(original, 0), [text(""), text("")])
  assert.equal(original[0].text, "a😀\u4e2d")
  assert.deepEqual(retainToolResultContent([text("\ud800x")], 3), [text("�")])
  assert.throws(() => retainToolResultContent(original, -1), /non-negative/)
  assert.throws(() => retainToolResultContent([{ type: "image", url: "x" }], 3), /text\/reasoning/)
})

test("retry identity is stable, native cutoff advances, and equal text sources stay distinct", () => {
  const { ledger } = makeLedger()
  const first = ledger.capture(request([message("a", "same"), message("b", "same")]), facts())
  ledger.observeNativeEvent("s", { seq: 12 })
  const retry = ledger.capture(request([message("a", "same"), message("b", "same")]), facts(2))
  assert.equal(first.status, "complete")
  assert.equal(first.snapshot, true)
  assert.equal(retry.snapshot, false)
  assert.equal(retry.request_hash, first.request_hash)
  assert.equal(retry.snapshot_cutoff_native_seq, 8)
  assert.equal(retry.cutoff_native_seq, 12)
  assert.notEqual(first.attempt_id, retry.attempt_id)
  assert.deepEqual(first.parts, retry.parts)
  const equal = first.parts.filter((part) => part.source_text === "same")
  assert.notEqual(equal[0].content_id, equal[1].content_id)
  const changed = ledger.capture(request([message("a", "changed")]), facts(3))
  assert.equal(changed.status, "missing")
  assert.ok(changed.missing_fields.includes("snapshot:request-hash-mismatch"))
})

test("missing and non-JSON request content is explicit", () => {
  const { ledger } = makeLedger()
  const missing = ledger.capture(request([{ role: "user", content: [text("x")] }]), facts())
  assert.equal(missing.status, "missing")
  assert.ok(missing.parts.some((part) => part.content_id === null))
  const unsupported = ledger.capture({ ...request([]), messages: { invalid: true }, extension: () => {} }, facts(1, 2))
  assert.equal(unsupported.status, "missing")
  assert.ok(unsupported.missing_fields.includes("request/extension:json-unavailable"))
  assert.ok(unsupported.missing_fields.includes("request/messages"))
})

test("tool policy preserves source ranges through message cloning and detects subsequent mutation", async () => {
  const policy = createToolResultContentPolicy({ maxUtf8Bytes: 5 })
  try {
    const decision = await policy({ callId: "call", agent: { session: { id: "s" } } },
      { content: [text("a😀\u4e2d")] }, async () => ({ kind: "accept" }))
    const toolMessage = { id: "tool-msg", role: "user", content: [{ type: "tool-result", toolCallId: "call", content: structuredClone(decision.content) }] }
    const { ledger } = makeLedger()
    const record = ledger.capture(request([toolMessage]), facts())
    const part = record.parts.find((item) => item.path.endsWith("/content/0/text"))
    assert.equal(record.status, "complete")
    assert.equal(part.source_text, "a😀\u4e2d")
    assert.equal(part.source_utf8_bytes, 8)
    assert.deepEqual(part.retained_ranges, [[0, 5]])
    assert.equal(record.request.messages[0].content[0].content[0].text, "a😀")
    toolMessage.content[0].content[0].text = "summary"
    const modified = ledger.capture(request([toolMessage]), facts(1, 2))
    assert.equal(modified.status, "missing")
    assert.ok(modified.missing_fields.some((field) => field.includes("retention-provenance-mismatch")))
  } finally { disposeRequestContentSources("s") }
})

let hostRequire
try {
  hostRequire = createRequire(join(process.env.DSH_HOME || join(homedir(), ".dsh"), "profiles", "package.json"))
  hostRequire.resolve("@deepseek-ai/dsh-agent-loop")
} catch { hostRequire = null }

test("installed DSH sends retained tool content and fsyncs every actual Attempt with stable retry identity", {
  skip: hostRequire === null ? "installed DSH runtime unavailable" : false,
}, async () => {
  const outputDir = await mkdtemp(join(tmpdir(), "request-content-ledger-"))
  const load = (name) => import(pathToFileURL(hostRequire.resolve(`@deepseek-ai/${name}`)).href)
  const [cordis, sessions, agents, prompts, tools, llm, loop, meter, plugin] = await Promise.all([
    load("cordis"), load("dsh-session"), load("dsh-agent"), load("dsh-system-prompt"),
    load("dsh-tools"), load("dsh-llm"), load("dsh-agent-loop"), load("dsh-token-meter"), import("../index.mjs"),
  ])
  const ctx = new cordis.Context()
  let handle
  const sent = []
  class Adapter extends llm.LlmAdapter {
    async resolveModel(provider, model) { return { provider, id: model, name: model, context: { contextWindow: 4096 } } }
    async *stream(options) {
      assert.equal(Object.isFrozen(options), true)
      sent.push(structuredClone({ messages: options.messages, tools: options.tools, system: options.system }))
      if ([1, 3, 4].includes(sent.length)) {
        yield { type: "block-start", index: 0, blockType: "tool-call" }
        yield { type: "block-end", index: 0, block: { type: "tool-call", id: llm.CallId(`call-${sent.length}`), name: "read_fixture", arguments: "{}" } }
        yield { type: "finish", reason: { kind: "tool-calls" } }
      } else if (sent.length === 2) {
        throw new llm.LlmError("retry fixture", "FIXTURE_RETRY")
      } else {
        yield { type: "finish", reason: { kind: "stop" } }
      }
    }
  }
  try {
    await ctx.plugin(sessions.default)
    await ctx.plugin(agents.default)
    await ctx.plugin(prompts.default, { includeHarnessIdentity: false, includeRuntimeContext: false, persona: "fixture" })
    await ctx.plugin(tools.default)
    await ctx.plugin(llm.default)
    await ctx.plugin(meter.default)
    await ctx.plugin(loop.default, { agents: [], maxParallelToolCalls: 1 })
    await ctx.plugin(plugin, { outputDir, requestContentLedger: true })
    ctx.llm.registerAdapter(["fixture"], new Adapter())
    ctx.tools.register({ name: "read_fixture", description: "fixture", parameters: { type: "object", properties: {} },
      output: { schema: { type: "string" }, render: (_args, value) => [text(value)] },
      execute: async () => "a😀\u4e2dtail",
    })
    handle = await ctx.agents.create({
      sessionId: sessions.SessionId("s"), meta: { cwd: outputDir }, agentOptions: { provider: "fixture", model: "fixture" },
      setup(agentCtx) {
        agentCtx.on("tools/post-execute", createToolResultContentPolicy({ maxUtf8Bytes: 5 }))
        registerRequestHistoryPolicy("s", 2)
        agentCtx.on("agent/pre-step", async (_payload, next) => {
          const decision = await next()
          if (decision.kind === "enter") pruneToolResultHistory(handle.agent.session, {
            keepRecent: 2, freezeMessage: llm.freezeMessage,
            estimateMessage: (message) => ctx.tokenMeter.estimateMessage(message),
          })
          return decision
        })
        agentCtx.on("agent/request-error", (payload) => payload.failure.code === "FIXTURE_RETRY" ? { kind: "retry" } : undefined)
      },
    })
    handle.agent.followup(llm.createUserMessage({ content: [text("read fixture")], source: { kind: "user" } }))
    await handle.agent.whenIdle()
    assert.equal(sent.length, 5, JSON.stringify(handle.agent.session.events.at(-1)))
    await ctx.sessions.flush(handle.agent.session)
    const records = (await readFile(join(outputDir, "request-content", "s.jsonl"), "utf8")).trim().split("\n").map(JSON.parse)
    assert.equal(records.length, 5)
    assert.ok(records.every((record) => record.status === "complete"), JSON.stringify(records.map((record) => record.missing_fields)))
    assert.deepEqual(records.map((record) => record.snapshot), [true, true, false, true, true])
    assert.equal(records[1].request_hash, records[2].request_hash)
    assert.notEqual(records[1].attempt_id, records[2].attempt_id)
    for (const [index, record] of records.entries()) {
      assert.equal(canonicalRequestJson(record.request.messages), canonicalRequestJson(sent[index].messages))
    }
    const actual = sent[1].messages.flatMap((item) => item.content).find((block) => block.type === "tool-result")
    assert.equal(actual.content[0].text, "a😀")
    const retained = records[1].parts.find((part) => part.source_text === "a😀\u4e2dtail")
    assert.deepEqual(retained.retained_ranges, [[0, 5]])
    const finalResults = sent[4].messages.flatMap((item) => item.content).filter((block) => block.type === "tool-result")
    assert.deepEqual(finalResults.map((block) => block.content), [[], [text("a😀")], [text("a😀")]])
    const removed = records[4].removed_parts.find((part) => part.content_id === retained.content_id)
    assert.deepEqual(removed.retained_ranges, [])
    assert.equal(records[4].history_keep_recent_tool_results, 2)
    assert.ok(records[4].retention.some((item) => Number.isInteger(item.history_replacement_seq)))
    assert.equal(handle.agent.session.events.filter((event) => event.type === "compaction/prune").length, 1)
    assert.deepEqual(handle.agent.session.deriveMessages().filter((item) => item.source.kind === "tool")
      .map((item) => item.content[0].content), [[], [text("a😀")], [text("a😀")]])
  } finally {
    await handle?.dispose()
    await ctx.fiber.dispose()
    disposeRequestContentSources("s")
    await rm(outputDir, { recursive: true, force: true })
  }
})
