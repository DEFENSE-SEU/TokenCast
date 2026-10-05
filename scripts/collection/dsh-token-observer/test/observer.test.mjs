import assert from "node:assert/strict"
import { mkdtemp, readFile, rm } from "node:fs/promises"
import { createRequire } from "node:module"
import { homedir, tmpdir } from "node:os"
import { join } from "node:path"
import { pathToFileURL } from "node:url"
import test from "node:test"

import {
  createStreamMiddleware,
  DshStreamObserver,
  generationProbeBridgeRegistry,
  OBSERVER_DURABILITY_CONTRACT_ID,
  OBSERVER_UNIT_CONTRACT_ID,
  requestSnapshot,
} from "../observer.mjs"
import {
  ObserverSidecarWriter,
  observerSidecarPath,
} from "../sidecar-writer.mjs"

function dshHomePath() {
  return process.env.DSH_HOME || join(homedir(), ".dsh")
}

function resolveInstalledDshLlm() {
  try {
    const profileRequire = createRequire(join(dshHomePath(), "profiles", "package.json"))
    return profileRequire.resolve("@deepseek-ai/dsh-llm")
  } catch {
    return null
  }
}

function request(overrides = {}) {
  return {
    provider: "fixture-provider",
    model: "fixture-model",
    messages: [{ role: "user", content: [{ type: "text", text: "repair it" }] }],
    system: "fixture system",
    tools: [],
    temperature: 0,
    maxTokens: 64,
    sessionId: "session-fixture",
    signal: new AbortController().signal,
    ...overrides,
  }
}

async function* chunks(items) {
  yield* items
}

async function drain(stream) {
  const values = []
  for await (const value of stream) values.push(value)
  return values
}

const installedDshLlm = resolveInstalledDshLlm()

test("sidecar writer flushes dirty records and closes each session", async () => {
  const outputDir = await mkdtemp(join(tmpdir(), "token-prediction-observer-writer-"))
  const calls = []
  try {
    const writer = new ObserverSidecarWriter({
      outputDir,
      openFile(path, flags) {
        calls.push(["open", path, flags])
        return 17
      },
      writeFile(fd, value) {
        calls.push(["write", fd, value])
      },
      syncFile(fd) {
        calls.push(["sync", fd])
      },
      closeFile(fd) {
        calls.push(["close", fd])
      },
    })
    writer.writeRecord("session-a", { record_type: "observer/session" })
    assert.equal(writer.flushSession("session-a"), true)
    assert.equal(writer.flushSession("session-a"), false)
    writer.writeRecord("session-a", { record_type: "attempt/start" })
    assert.equal(writer.closeSession("session-a"), true)
    assert.equal(writer.closeSession("session-a"), false)
    assert.deepEqual(calls, [
      ["open", observerSidecarPath(outputDir, "session-a"), "ax"],
      ["write", 17, '{"record_type":"observer/session"}\n'],
      ["sync", 17],
      ["write", 17, '{"record_type":"attempt/start"}\n'],
      ["sync", 17],
      ["close", 17],
    ])
  } finally {
    await rm(outputDir, { recursive: true, force: true })
  }
})

test("sidecar writer retains a descriptor until a failed close can be retried", async () => {
  const outputDir = await mkdtemp(join(tmpdir(), "token-prediction-observer-close-"))
  const failure = new Error("fixture close failure")
  let closeCalls = 0
  try {
    const writer = new ObserverSidecarWriter({
      outputDir,
      openFile() {
        return 23
      },
      writeFile() {},
      syncFile() {},
      closeFile(fd) {
        assert.equal(fd, 23)
        closeCalls += 1
        if (closeCalls === 1) throw failure
      },
    })
    writer.writeRecord("session-a", { record_type: "observer/session" })

    assert.throws(() => writer.closeSession("session-a"), failure)
    assert.equal(writer.sessions.has("session-a"), true)
    writer.closeAll()

    assert.equal(closeCalls, 2)
    assert.equal(writer.sessions.has("session-a"), false)
  } finally {
    await rm(outputDir, { recursive: true, force: true })
  }
})

test("sidecar writer preserves strict JSONL and rejects a reused session file", async () => {
  const outputDir = await mkdtemp(join(tmpdir(), "token-prediction-observer-file-"))
  try {
    const writer = new ObserverSidecarWriter({ outputDir })
    writer.writeRecord("session-a", { session_id: "session-a", value: 1 })
    writer.writeRecord("session-a", { session_id: "session-a", value: 2 })
    writer.flushSession("session-a")
    writer.closeSession("session-a")
    const path = observerSidecarPath(outputDir, "session-a")
    assert.equal(
      await readFile(path, "utf8"),
      '{"session_id":"session-a","value":1}\n'
        + '{"session_id":"session-a","value":2}\n',
    )
    assert.throws(
      () => writer.writeRecord("session-a", { session_id: "session-a", value: 3 }),
      { code: "EEXIST" },
    )
  } finally {
    await rm(outputDir, { recursive: true, force: true })
  }
})

test("sidecar writer poisons a session after a record write fails", async () => {
  const outputDir = await mkdtemp(join(tmpdir(), "token-prediction-observer-failure-"))
  const failure = new Error("fixture partial record")
  const calls = []
  try {
    const writer = new ObserverSidecarWriter({
      outputDir,
      openFile() {
        return 19
      },
      writeFile() {
        calls.push("write")
        throw failure
      },
      syncFile() {
        calls.push("sync")
      },
      closeFile() {
        calls.push("close")
      },
    })
    assert.throws(
      () => writer.writeRecord("session-a", { record_type: "attempt/start" }),
      (error) => error === failure,
    )
    assert.throws(
      () => writer.writeRecord("session-a", { record_type: "attempt/end" }),
      (error) => error === failure,
    )
    assert.throws(
      () => writer.closeSession("session-a"),
      (error) => error === failure,
    )
    assert.deepEqual(calls, ["write", "sync", "close"])
  } finally {
    await rm(outputDir, { recursive: true, force: true })
  }
})

test(
  "linked plugin loads and shares the running DSH agent-loop request gate",
  { skip: installedDshLlm === null ? "an installed DSH runtime is unavailable" : false },
  async () => {
    const outputDir = await mkdtemp(join(tmpdir(), "token-prediction-dsh-observer-"))
    let ctx
    try {
      const plugin = await import("../index.mjs")
      const hostDshLlm = await import(pathToFileURL(installedDshLlm).href)
      const hostSession = await import(plugin.resolveDshHostModuleUrl("@deepseek-ai/dsh-session"))
      const hostCordis = await import(plugin.resolveDshHostModuleUrl("@deepseek-ai/cordis"))
      assert.equal(
        plugin.resolveDshHostModuleUrl("@deepseek-ai/dsh-llm"),
        pathToFileURL(installedDshLlm).href,
      )

      ctx = new hostCordis.Context()
      await ctx.plugin(hostDshLlm.LlmRuntime)
      await ctx.plugin(hostSession.SessionStore)
      await ctx.plugin(plugin, { outputDir, checkpointIntervalBytes: 4 })
      const session = ctx.sessions.create("session-fixture", {
        meta: { cwd: outputDir, agentPreset: "router-standard" },
      })
      ctx.emit(
        "session/event",
        session,
        { type: "step/start", data: { turn: 0, step: 0 } },
      )

      const marked = hostDshLlm.markAgentLoopRequest(request())
      await drain(
        ctx.waterfall(
          ctx.llm,
          "llm/stream",
          marked,
          () => chunks([{ type: "finish", reason: { kind: "stop" } }]),
        ),
      )
      await drain(
        ctx.waterfall(
          ctx.llm,
          "llm/stream",
          request(),
          () => chunks([{ type: "finish", reason: { kind: "stop" } }]),
        ),
      )
      assert.equal(await ctx.sessions.flush(session), true)

      const sidecar = await readFile(join(outputDir, "session-fixture.jsonl"), "utf8")
      const records = sidecar
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line))
      assert.deepEqual(
        records.map((record) => record.record_type),
        ["observer/session", "attempt/start", "attempt/end"],
      )
      assert.equal(records[1].attempt_ordinal, 1)
      assert.equal(records[1].schema_version, 2)
      assert.equal(records[1].request_id, "session-fixture:turn-0:step-0:request")
      assert.equal(records[2].finish_reason, "stop")
      assert.equal(
        records[0].observer_durability_contract_id,
        OBSERVER_DURABILITY_CONTRACT_ID,
      )
    } finally {
      await ctx?.fiber.dispose()
      await rm(outputDir, { recursive: true, force: true })
    }
  },
)

test(
  "session store awaits observer sync failures and disposal closes the writer",
  { skip: installedDshLlm === null ? "an installed DSH runtime is unavailable" : false },
  async () => {
    const plugin = await import("../index.mjs")
    const hostSession = await import(plugin.resolveDshHostModuleUrl("@deepseek-ai/dsh-session"))
    const hostCordis = await import(plugin.resolveDshHostModuleUrl("@deepseek-ai/cordis"))
    const calls = []
    const ctx = new hostCordis.Context()
    await ctx.plugin(hostSession.SessionStore)
    let failSync = false
    const sidecars = {
      writeRecord() {},
      flushSession(sessionId) {
        calls.push(["flush", sessionId])
        if (failSync) throw new Error("fixture observer sync failure")
      },
      closeSession(sessionId) {
        calls.push(["close", sessionId])
      },
      closeAll() {
        calls.push(["close-all"])
      },
    }
    plugin.installObserver(ctx, { checkpointIntervalBytes: 4 }, sidecars)
    const session = ctx.sessions.create("session-a", { meta: { cwd: tmpdir() } })
    assert.equal(await ctx.sessions.flush(session), true)
    failSync = true
    await assert.rejects(ctx.sessions.flush(session), /fixture observer sync failure/)
    await ctx.fiber.dispose()
    assert.deepEqual(calls, [
      ["flush", "session-a"],
      ["flush", "session-a"],
      ["close", "session-a"],
      ["close-all"],
    ])
  },
)

test("request snapshots omit live signal and session routing identity", () => {
  const first = request()
  const second = request({
    sessionId: "another-session",
    signal: new AbortController().signal,
  })
  assert.deepEqual(requestSnapshot(first), requestSnapshot(second))
  assert.notDeepEqual(requestSnapshot(first), requestSnapshot(request({ maxTokens: 65 })))
})

test("middleware filters auxiliary calls and invokes downstream exactly once", async () => {
  const records = []
  const observer = new DshStreamObserver({
    checkpointIntervalBytes: 8,
    writeRecord: (_sessionId, record) => records.push(record),
  })
  const marked = new WeakSet()
  const agentRequest = request()
  marked.add(agentRequest)
  const middleware = createStreamMiddleware({
    observer,
    isAgentLoopRequest: (value) => marked.has(value),
    activeStep: () => ({ turn: 0, step: 0 }),
  })
  let calls = 0
  const next = () => {
    calls += 1
    return chunks([{ type: "finish", reason: { kind: "stop" } }])
  }
  const auxiliary = request()
  await drain(middleware(auxiliary, next))
  assert.equal(calls, 1)
  assert.equal(records.length, 0)
  await drain(middleware(agentRequest, next))
  assert.equal(calls, 2)
  assert.equal(records.filter((item) => item.record_type === "attempt/start").length, 1)
})

test("observer emits attempt-scoped chunk checkpoints and excludes hidden streams", async () => {
  const records = []
  let now = 1_780_000_000_000
  const observer = new DshStreamObserver({
    checkpointIntervalBytes: 4,
    now: () => now++,
    writeRecord: (_sessionId, record) => records.push(record),
  })
  const source = [
    { type: "reasoning-delta", index: 0, text: "hidden" },
    { type: "text-delta", index: 1, text: "ab" },
    { type: "tool-call-delta", index: 2, id: "call-1", argumentsDelta: "private" },
    { type: "text-delta", index: 1, text: "cd" },
    { type: "text-delta", index: 1, text: "\ud83d" },
    { type: "text-delta", index: 1, text: "\ude80!" },
    {
      type: "usage",
      usage: {
        inputTokens: 10,
        outputTokens: 7,
        cacheReadTokens: 2,
        reasoningTokens: 3,
      },
    },
    { type: "finish", reason: { kind: "stop" } },
  ]
  assert.deepEqual(
    await drain(observer.observe(request(), chunks(source), { turn: 0, step: 1 })),
    source,
  )

  assert.equal(records[0].record_type, "observer/session")
  assert.equal(
    records[0].observer_durability_contract_id,
    OBSERVER_DURABILITY_CONTRACT_ID,
  )
  const start = records.find((item) => item.record_type === "attempt/start")
  const checkpoints = records.filter((item) => item.record_type === "attempt/checkpoint")
  const end = records.find((item) => item.record_type === "attempt/end")
  assert.equal(start.attempt_ordinal, 1)
  assert.equal(start.retry_of_attempt_id, null)
  assert.deepEqual(
    checkpoints.map((item) => [item.checkpoint_ordinal, item.observer_committed_tokens]),
    [[1, 4], [2, 9]],
  )
  assert.equal(checkpoints.map((item) => item.visible_text_delta).join(""), "abcd🚀!")
  assert.equal(checkpoints[0].observer_unit_contract_id, OBSERVER_UNIT_CONTRACT_ID)
  assert.equal(end.observer_visible_output_tokens, 9)
  assert.equal(end.observer_finalized, true)
  assert.equal(end.visible_text_tail, "")
  assert.deepEqual(end.usage, {
    inputTokens: 10,
    outputTokens: 7,
    cacheReadTokens: 2,
    cacheWriteTokens: null,
    reasoningTokens: 3,
  })
})

test("observer commits each chunk only after the consumer resumes the stream", async () => {
  const records = []
  const observer = new DshStreamObserver({
    checkpointIntervalBytes: 4,
    writeRecord: (_sessionId, record) => records.push(record),
  })
  const stream = observer
    .observe(
      request(),
      chunks([
        { type: "text-delta", index: 0, text: "abcd" },
        {
          type: "usage",
          usage: { inputTokens: 5, outputTokens: 1 },
        },
        { type: "finish", reason: { kind: "stop" } },
      ]),
      { turn: 0, step: 0 },
    )
    [Symbol.asyncIterator]()

  assert.equal((await stream.next()).value.type, "text-delta")
  assert.deepEqual(
    records.map((item) => item.record_type),
    ["observer/session", "attempt/start"],
  )

  assert.equal((await stream.next()).value.type, "usage")
  assert.deepEqual(
    records.map((item) => item.record_type),
    ["observer/session", "attempt/start", "attempt/checkpoint"],
  )

  assert.equal((await stream.next()).value.type, "finish")
  assert.equal(records.some((item) => item.record_type === "attempt/end"), false)

  assert.deepEqual(await stream.next(), { value: undefined, done: true })
  const end = records.find((item) => item.record_type === "attempt/end")
  assert.equal(end.finish_reason, "stop")
  assert.equal(end.observer_finalized, true)
  assert.equal(end.observer_visible_output_tokens, 4)
  assert.deepEqual(end.usage, {
    inputTokens: 5,
    outputTokens: 1,
    cacheReadTokens: null,
    cacheWriteTokens: null,
    reasoningTokens: null,
  })
})

test("Generation Probe receives the committed checkpoint before source acknowledgement", async () => {
  const records = []
  const timeline = []
  const observer = new DshStreamObserver({
    checkpointIntervalBytes: 2,
    writeRecord: (_sessionId, record) => {
      records.push(record)
      if (record.record_type === "attempt/checkpoint") {
        timeline.push(`observer-write-${record.checkpoint_ordinal}`)
      }
    },
  })
  const bridges = [{ checkpoints: [] }, { checkpoints: [] }]
  const probeChunk = (text, bridge) => {
    const chunk = { type: "text-delta", index: 0, text }
    generationProbeBridgeRegistry().set(chunk, bridge)
    return chunk
  }
  async function* source() {
    timeline.push("sample-1")
    yield probeChunk("ab", bridges[0])
    timeline.push("ack-1")
    timeline.push("sample-2")
    yield probeChunk("cd", bridges[1])
    timeline.push("ack-2")
    yield { type: "usage", usage: { inputTokens: 4, outputTokens: 2 } }
    yield { type: "finish", reason: { kind: "stop" } }
  }

  const stream = observer.observe(request(), source(), { turn: 3, step: 4 })[Symbol.asyncIterator]()
  const first = await stream.next()
  assert.equal(first.value.text, "ab")
  assert.deepEqual(bridges[0].checkpoints, [])
  assert.equal(JSON.stringify(first.value), '{"type":"text-delta","index":0,"text":"ab"}')

  const second = await stream.next()
  assert.equal(second.value.text, "cd")
  assert.deepEqual(timeline.slice(0, 4), [
    "sample-1",
    "observer-write-1",
    "ack-1",
    "sample-2",
  ])
  assert.deepEqual(bridges[0].checkpoints, [
    {
      session_id: "session-fixture",
      attempt_id: "session-fixture:turn-3:step-4:attempt-1",
      checkpoint_ordinal: 1,
      observer_committed_tokens: 2,
    },
  ])
  assert.deepEqual(bridges[1].checkpoints, [])

  assert.equal((await stream.next()).value.type, "usage")
  assert.deepEqual(timeline.slice(4), ["observer-write-2", "ack-2"])
  assert.deepEqual(bridges[1].checkpoints, [
    {
      session_id: "session-fixture",
      attempt_id: "session-fixture:turn-3:step-4:attempt-1",
      checkpoint_ordinal: 2,
      observer_committed_tokens: 4,
    },
  ])
  assert.equal((await stream.next()).value.type, "finish")
  assert.deepEqual(await stream.next(), { value: undefined, done: true })
  assert.equal(
    records.filter((record) => record.record_type === "attempt/checkpoint").length,
    2,
  )
})

test("consumer cancellation excludes the unacknowledged chunk", async () => {
  const records = []
  const observer = new DshStreamObserver({
    checkpointIntervalBytes: 1,
    writeRecord: (_sessionId, record) => records.push(record),
  })
  const unopened = observer
    .observe(request(), chunks([{ type: "text-delta", index: 0, text: "x" }]), {
      turn: 0,
      step: 0,
    })
    [Symbol.asyncIterator]()
  assert.deepEqual(await unopened.return(), { value: undefined, done: true })
  assert.equal(records.length, 0)

  const stream = observer
    .observe(request(), chunks([{ type: "text-delta", index: 0, text: "x" }]), {
      turn: 0,
      step: 0,
    })
    [Symbol.asyncIterator]()
  assert.equal((await stream.next()).value.text, "x")
  assert.equal(records.some((item) => item.record_type === "attempt/checkpoint"), false)
  await stream.return()

  const end = records.find((item) => item.record_type === "attempt/end")
  assert.equal(records.some((item) => item.record_type === "attempt/checkpoint"), false)
  assert.equal(end.finish_reason, "consumer-cancelled")
  assert.equal(end.observer_finalized, false)
  assert.equal(end.observer_visible_output_tokens, 0)
  assert.equal(end.visible_text_tail, "")
  assert.equal(end.usage, null)
})

test("consumer exceptions and source exceptions have distinct terminals", async () => {
  const consumerRecords = []
  const consumerObserver = new DshStreamObserver({
    checkpointIntervalBytes: 1,
    writeRecord: (_sessionId, record) => consumerRecords.push(record),
  })
  const consumerStream = consumerObserver
    .observe(request(), chunks([{ type: "text-delta", index: 0, text: "x" }]), {
      turn: 0,
      step: 0,
    })
    [Symbol.asyncIterator]()
  await consumerStream.next()
  await assert.rejects(() => consumerStream.throw(new Error("consumer failed")), /consumer failed/)
  const consumerEnd = consumerRecords.find((item) => item.record_type === "attempt/end")
  assert.equal(consumerEnd.finish_reason, "consumer-exception")
  assert.equal(consumerEnd.observer_visible_output_tokens, 0)

  const sourceRecords = []
  const sourceObserver = new DshStreamObserver({
    checkpointIntervalBytes: 1,
    writeRecord: (_sessionId, record) => sourceRecords.push(record),
  })
  async function* brokenBeforeFirstChunk() {
    throw new Error("source failed")
  }
  await assert.rejects(
    () => drain(sourceObserver.observe(request(), brokenBeforeFirstChunk(), { turn: 1, step: 0 })),
    /source failed/,
  )
  const sourceEnd = sourceRecords.find((item) => item.record_type === "attempt/end")
  assert.equal(sourceEnd.finish_reason, "stream-exception")
  assert.equal(sourceEnd.observer_visible_output_tokens, 0)
})

test("every stream in one DSH step remains an ordered Attempt of one Call", async () => {
  const records = []
  const observer = new DshStreamObserver({
    checkpointIntervalBytes: 32,
    writeRecord: (_sessionId, record) => records.push(record),
  })
  const failed = chunks([
    { type: "finish", reason: { kind: "error", failure: { code: "RATE_LIMIT" } } },
  ])
  const completed = chunks([{ type: "finish", reason: { kind: "stop" } }])
  const step = { turn: 2, step: 3 }
  await drain(observer.observe(request(), failed, step))
  await drain(observer.observe(request(), completed, step))
  await assert.rejects(
    () => drain(observer.observe(request({ model: "fixture-model-2" }), completed, step)),
    /retry changed the logical Call request/,
  )

  const starts = records.filter((item) => item.record_type === "attempt/start")
  assert.equal(starts[0].logical_call_id, starts[1].logical_call_id)
  assert.equal(starts[1].attempt_ordinal, 2)
  assert.equal(starts[1].retry_of_attempt_id, starts[0].attempt_id)
  assert.equal(starts[0].request_id, "session-fixture:turn-2:step-3:request")
  assert.equal(starts[1].request_id, starts[0].request_id)
})

test("an exceptional stream closes observer accounting as unfinalized", async () => {
  const records = []
  const observer = new DshStreamObserver({
    checkpointIntervalBytes: 1,
    writeRecord: (_sessionId, record) => records.push(record),
  })
  async function* broken() {
    yield { type: "text-delta", index: 0, text: "x" }
    throw new Error("fixture failure")
  }
  await assert.rejects(
    () => drain(observer.observe(request(), broken(), { turn: 4, step: 5 })),
    /fixture failure/,
  )
  const end = records.find((item) => item.record_type === "attempt/end")
  assert.equal(end.finish_reason, "stream-exception")
  assert.equal(end.observer_finalized, false)
  assert.equal(end.observer_visible_output_tokens, 1)
})

test("tool arguments count toward checkpoints only when asked for", async () => {
  const source = [
    { type: "tool-call-delta", index: 0, id: "call-1", argumentsDelta: "12345678" },
    { type: "tool-call-delta", index: 0, id: "call-1", argumentsDelta: "90abcdef" },
    { type: "finish", reason: { kind: "stop" } },
  ]
  const run = async (options) => {
    const records = []
    let now = 1_780_000_000_000
    const observer = new DshStreamObserver({
      checkpointIntervalBytes: 4,
      now: () => now++,
      writeRecord: (_sessionId, record) => records.push(record),
      ...options,
    })
    await drain(observer.observe(request(), chunks(source), { turn: 0, step: 1 }))
    return records.filter((item) => item.record_type === "attempt/checkpoint")
  }

  assert.equal((await run({})).length, 0, "default policy must ignore tool arguments")

  const counted = await run({ countToolArguments: true })
  assert.ok(counted.length >= 2, "opting in must produce checkpoints")
  assert.ok(
    counted.every((item) => item.observer_committed_tokens > 0),
    "each checkpoint carries the bytes committed so far",
  )
})
