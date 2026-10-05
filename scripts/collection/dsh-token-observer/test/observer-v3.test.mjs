import assert from "node:assert/strict"
import test from "node:test"

import {
  DshGenerationObserver,
  GENERATION_CHANNELS,
  NativeSessionIndex,
  generationCheckpointPolicyId,
  splitCompleteText,
} from "../observer-v3.mjs"

function clock(start = 1781000000000) {
  let time = start
  return () => (time += 1)
}

function measureTokens(_session, requestHeader) {
  return {
    logRevision: 5,
    baseline: { kind: requestHeader === undefined ? "none" : "estimated", tokens: requestHeader === undefined ? 0 : 55 },
    surfaceDeltaTokens: 0,
    totalTokens: requestHeader === undefined ? 0 : 55,
    surfaceTokens: 0,
    nodes: [],
  }
}

function makeObserver({ checkpointIntervalBytes = 4 } = {}) {
  const records = []
  const observer = new DshGenerationObserver({
    checkpointIntervalBytes,
    writeRecord: (_sessionId, record) => records.push(record),
    now: clock(),
    measureTokens,
  })
  return { observer, records }
}

function seedNativeEvents(observer, sessionId = "s1") {
  const events = [
    [3, "step/start", { turn: 1, step: 1 }],
    [4, "user/message", { role: "user" }],
    [5, "request/header", { header: { config: {} }, reason: "initial" }],
    [6, "request/context", { provider: "p", model: "m", contextWindow: 4096 }],
  ]
  for (const [seq, type, data] of events) {
    observer.observeNativeEvent(sessionId, { seq, type, data })
  }
}

function request(overrides = {}) {
  return {
    sessionId: "s1",
    provider: "p",
    model: "m",
    messages: [{ content: [1, 2] }],
    tools: [{ name: "shell" }],
    stop: [],
    ...overrides,
  }
}

async function drain(observer, chunks, step = { turn: 1, step: 1 }) {
  const out = []
  for await (const chunk of observer.observe(request(), (async function* () {
    for (const item of chunks) yield item
  })(), step)) {
    out.push(chunk)
  }
  return out
}

test("the surrogate carry never counts a half pair", () => {
  const first = splitCompleteText("", "ab\ud83d")
  assert.equal(first.complete, "ab")
  assert.equal(first.carry, "\ud83d")
  const second = splitCompleteText(first.carry, "\ude00")
  assert.equal(second.complete, "ab😀".slice(2))
  assert.equal(second.carry, "")
})

test("the native index keeps the seq the session service stamped", () => {
  const index = new NativeSessionIndex()
  index.observe({ seq: 5, type: "request/header", data: { a: 1 } })
  index.observe({ seq: 6, type: "request/context", data: { b: 2 } })
  assert.deepEqual(index.latest("request/header"), { native_seq: 5, value: { a: 1 } })
  assert.deepEqual(index.latest("request/context"), { native_seq: 6, value: { b: 2 } })
  assert.equal(index.latest("missing"), null)
})

test("message nodes accumulate into the projection reference", () => {
  const index = new NativeSessionIndex()
  index.observe({ seq: 4, type: "user/message", data: {} })
  index.observe({ seq: 9, type: "assistant/message", data: {} })
  index.observe({ seq: 9, type: "assistant/message", data: {} })
  assert.deepEqual(index.messageRefs(), { native_seqs: [4, 9] })
})

test("the session record declares the generation unit and its channels", () => {
  const { observer } = makeObserver()
  const record = observer.sessionRecord("s1", {
    dshRelease: "0.1.1-rc.2",
    dshRevision: "abc",
  })
  assert.equal(record.schema_version, 3)
  assert.equal(record.generation_unit_contract_id, "observer:generation-utf8-bytes-v2")
  assert.deepEqual(record.generation_channels, GENERATION_CHANNELS)
  assert.equal(record.checkpoint_policy_id, generationCheckpointPolicyId(4))
  assert.equal(record.dsh_revision, "abc")
})

test("a call emits once per step and carries the native request seqs", async () => {
  const { observer, records } = makeObserver()
  seedNativeEvents(observer)
  await drain(observer, [{ type: "finish", reason: { kind: "stop" } }])
  await drain(observer, [{ type: "finish", reason: { kind: "stop" } }])
  const calls = records.filter((record) => record.record_type === "call/start")
  assert.equal(calls.length, 1)
  assert.deepEqual(calls[0].request_header, {
    native_seq: 5,
    value: { header: { config: {} }, reason: "initial" },
  })
  assert.equal(calls[0].request_context.native_seq, 6)
  assert.equal(calls[0].effective_request_projection.messages_ref.native_seqs[0], 4)
})

test("all three generation channels are counted separately", async () => {
  const { observer, records } = makeObserver({ checkpointIntervalBytes: 1000 })
  seedNativeEvents(observer)
  await drain(observer, [
    { type: "text-delta", text: "hello" },
    { type: "reasoning-delta", text: "think" },
    { type: "tool-call-delta", argumentsDelta: '{"a":1}' },
    { type: "finish", reason: { kind: "tool-calls" } },
  ])
  const end = records.find((record) => record.record_type === "attempt/end")
  assert.deepEqual(end.channel_bytes, { text: 5, reasoning: 5, tool_arguments: 7 })
  assert.equal(end.generation_bytes, 17)
  assert.deepEqual(end.observed_channel_mask, {
    text: true,
    reasoning: true,
    tool_arguments: true,
  })
})

test("reasoning alone advances the generation clock", async () => {
  const { observer, records } = makeObserver({ checkpointIntervalBytes: 4 })
  seedNativeEvents(observer)
  await drain(observer, [
    { type: "reasoning-delta", text: "abcdefgh" },
    { type: "finish", reason: { kind: "stop" } },
  ])
  const checkpoints = records.filter((r) => r.record_type === "attempt/checkpoint")
  assert.equal(checkpoints.length, 1)
  assert.equal(checkpoints[0].generation_bytes, 8)
  assert.deepEqual(checkpoints[0].seen_channel_mask, {
    text: false,
    reasoning: true,
    tool_arguments: false,
  })
})

test("checkpoint ordinals and stream ordinals both advance", async () => {
  const { observer, records } = makeObserver({ checkpointIntervalBytes: 4 })
  seedNativeEvents(observer)
  await drain(observer, [
    { type: "text-delta", text: "aaaa" },
    { type: "text-delta", text: "bbbb" },
    { type: "text-delta", text: "cccc" },
    { type: "finish", reason: { kind: "stop" } },
  ])
  const checkpoints = records.filter((r) => r.record_type === "attempt/checkpoint")
  assert.deepEqual(
    checkpoints.map((c) => [c.checkpoint_ordinal, c.stream_event_ordinal, c.generation_bytes]),
    [
      [1, 1, 4],
      [2, 2, 8],
      [3, 3, 12],
    ],
  )
})

test("a retry attempt records its native retry positions", async () => {
  const { observer, records } = makeObserver()
  seedNativeEvents(observer)
  await drain(observer, [{ type: "finish", reason: { kind: "error", failure: { code: "E" } } }])
  observer.observeNativeEvent("s1", {
    seq: 12,
    type: "llm/retry",
    data: { retryId: "retry-1", turn: 1, step: 1, retry: 1, delayMs: 10 },
  })
  observer.observeNativeEvent("s1", { seq: 13, type: "llm/retry-started", data: {} })
  await drain(observer, [{ type: "finish", reason: { kind: "stop" } }])

  const starts = records.filter((record) => record.record_type === "attempt/start")
  assert.equal(starts.length, 2)
  assert.equal(starts[0].retry_id, null)
  assert.equal(starts[0].retry_ordinal, null)
  assert.equal(starts[1].retry_id, "retry-1")
  assert.equal(starts[1].retry_ordinal, 1)
  assert.equal(starts[1].retry_scheduled_native_seq, 12)
  assert.equal(starts[1].retry_started_native_seq, 13)
  assert.equal(starts[1].retry_of_attempt_id, starts[0].attempt_id)
})

test("an unfinished stream closes as unfinalized", async () => {
  const { observer, records } = makeObserver()
  seedNativeEvents(observer)
  await drain(observer, [{ type: "text-delta", text: "abc" }])
  const end = records.find((record) => record.record_type === "attempt/end")
  assert.equal(end.observer_finalized, false)
  assert.equal(end.finish_reason, "missing-finish")
})

test("a throwing stream still writes a terminal and rethrows", async () => {
  const { observer, records } = makeObserver()
  seedNativeEvents(observer)
  async function* broken() {
    yield { type: "text-delta", text: "ab" }
    throw new Error("boom")
  }
  await assert.rejects(async () => {
    for await (const _chunk of observer.observe(request(), broken(), { turn: 1, step: 1 })) {
    }
  })
  const end = records.find((record) => record.record_type === "attempt/end")
  assert.equal(end.observer_finalized, false)
  assert.equal(end.failure_code, "STREAM_EXCEPTION")
})

test("provider usage is carried onto the terminal", async () => {
  const { observer, records } = makeObserver()
  seedNativeEvents(observer)
  await drain(observer, [
    { type: "usage", usage: { inputTokens: 10, outputTokens: 2, cacheReadTokens: 3 } },
    { type: "finish", reason: { kind: "stop" } },
  ])
  const end = records.find((record) => record.record_type === "attempt/end")
  assert.equal(end.last_provider_usage.inputTokens, 10)
  assert.equal(end.last_provider_usage.outputTokens, 2)
  assert.equal(end.last_provider_usage.cacheReadTokens, 3)
})

test("a surrogate pair split across chunks counts four bytes once", async () => {
  const { observer, records } = makeObserver({ checkpointIntervalBytes: 1000 })
  seedNativeEvents(observer)
  await drain(observer, [
    { type: "text-delta", text: "\ud83d" },
    { type: "text-delta", text: "\ude00" },
    { type: "finish", reason: { kind: "stop" } },
  ])
  const end = records.find((record) => record.record_type === "attempt/end")
  assert.equal(end.channel_bytes.text, 4)
})

test("sessions are isolated and disposable", () => {
  const { observer } = makeObserver()
  observer.observeNativeEvent("a", { seq: 1, type: "request/header", data: {} })
  observer.observeNativeEvent("b", { seq: 2, type: "request/header", data: {} })
  assert.equal(observer.stateFor("a").native.latest("request/header").native_seq, 1)
  assert.equal(observer.stateFor("b").native.latest("request/header").native_seq, 2)
  observer.disposeSession("a")
  assert.equal(observer.stateFor("a").native.latest("request/header"), null)
})

test("the constructor rejects a nonsense checkpoint interval", () => {
  assert.throws(
    () => new DshGenerationObserver({ checkpointIntervalBytes: 0, writeRecord: () => {} }),
    TypeError,
  )
  assert.throws(() => new DshGenerationObserver({ writeRecord: null }), TypeError)
})

test("a native retry is republished as scheduled and started records", async () => {
  const { observer, records } = makeObserver()
  seedNativeEvents(observer)
  await drain(observer, [{ type: "finish", reason: { kind: "error", failure: { code: "E" } } }])
  observer.observeNativeEvent("s1", {
    seq: 12,
    type: "llm/retry",
    data: {
      retryId: "retry-1",
      turn: 1,
      step: 1,
      provider: "p",
      mode: "normal",
      policyKey: "key",
      retry: 1,
      delayMs: 10,
      failure: { code: "E", status: 503 },
    },
  })
  observer.observeNativeEvent("s1", {
    seq: 13,
    type: "llm/retry-started",
    data: { retryId: "retry-1", turn: 1, step: 1, retry: 1 },
  })

  const scheduled = records.find((r) => r.record_type === "retry/scheduled")
  const started = records.find((r) => r.record_type === "retry/started")
  assert.ok(scheduled, "a scheduled retry record is emitted")
  assert.equal(scheduled.native_seq, 12)
  assert.equal(scheduled.retry_id, "retry-1")
  assert.equal(scheduled.failure_code, "E")
  assert.equal(scheduled.failure_status, 503)
  assert.equal(scheduled.delay_ms, 10)
  assert.ok(scheduled.failed_attempt_id.endsWith("attempt-1"))
  assert.ok(started, "a started retry record is emitted")
  assert.equal(started.native_seq, 13)
  assert.equal(started.scheduled_native_seq, 12)
})

test("a retry outside any observed call is not republished", () => {
  const { observer, records } = makeObserver()
  observer.observeNativeEvent("s1", {
    seq: 4,
    type: "llm/retry",
    data: { retryId: "r", turn: 9, step: 9, retry: 1 },
  })
  assert.equal(records.filter((r) => r.record_type === "retry/scheduled").length, 0)
})

test("the observer refuses to record a request without the harness meter", async () => {
  const records = []
  const observer = new DshGenerationObserver({
    checkpointIntervalBytes: 4,
    writeRecord: (_sessionId, record) => records.push(record),
    now: clock(),
  })
  observer.observeNativeEvent("s1", { seq: 3, type: "step/start", data: { turn: 1, step: 1 } })
  await assert.rejects(async () => {
    for await (const _chunk of observer.observe(
      request(),
      (async function* () {
        yield { type: "finish", reason: { kind: "stop" } }
      })(),
      { turn: 1, step: 1 },
    )) {
    }
  }, /requires the harness token meter/)
})

test("the meter measurement is carried onto the request records", async () => {
  const { observer, records } = makeObserver()
  seedNativeEvents(observer)
  await drain(observer, [{ type: "finish", reason: { kind: "stop" } }])
  const call = records.find((record) => record.record_type === "call/start")
  assert.equal(call.token_meter.baseline.kind, "estimated")
  assert.equal(call.token_meter.baseline.tokens, 55)
  assert.equal(call.token_meter.logRevision, 5)
})

test("stream ordinals count every chunk, not only the ones carrying bytes", async () => {
  const { observer, records } = makeObserver({ checkpointIntervalBytes: 1000 })
  seedNativeEvents(observer)
  await drain(observer, [
    { type: "text-delta", text: "ab" },
    { type: "usage", usage: { inputTokens: 1, outputTokens: 1 } },
    { type: "finish", reason: { kind: "stop" } },
  ])
  const end = records.find((record) => record.record_type === "attempt/end")
  assert.equal(end.final_stream_event_ordinal, 3)
})

test("a retry that reuses its call request is not reported as changed", async () => {
  const { observer, records } = makeObserver()
  seedNativeEvents(observer)
  await drain(observer, [{ type: "finish", reason: { kind: "error", failure: { code: "E" } } }])
  observer.observeNativeEvent("s1", {
    seq: 12,
    type: "llm/retry",
    data: { retryId: "retry-1", turn: 1, step: 1, retry: 1 },
  })
  observer.observeNativeEvent("s1", {
    seq: 13,
    type: "llm/retry-started",
    data: { retryId: "retry-1", turn: 1, step: 1, retry: 1 },
  })
  await drain(observer, [{ type: "finish", reason: { kind: "stop" } }])
  const starts = records.filter((record) => record.record_type === "attempt/start")
  assert.equal(starts[1].request_changed_from_initial_call, false)
})

test("a retry whose request changed is reported as changed", async () => {
  const { observer, records } = makeObserver()
  seedNativeEvents(observer)
  await drain(observer, [{ type: "finish", reason: { kind: "error", failure: { code: "E" } } }])
  observer.observeNativeEvent("s1", {
    seq: 12,
    type: "llm/retry",
    data: { retryId: "retry-1", turn: 1, step: 1, retry: 1 },
  })
  observer.observeNativeEvent("s1", {
    seq: 13,
    type: "llm/retry-started",
    data: { retryId: "retry-1", turn: 1, step: 1, retry: 1 },
  })
  observer.observeNativeEvent("s1", {
    seq: 14,
    type: "request/header",
    data: { header: { config: { maxTokens: 128 } }, reason: "retry" },
  })
  await drain(observer, [{ type: "finish", reason: { kind: "stop" } }])
  const starts = records.filter((record) => record.record_type === "attempt/start")
  assert.equal(starts[1].request_changed_from_initial_call, true)
})

test("the call records the status the stream middleware can attest to", async () => {
  const { observer, records } = makeObserver()
  seedNativeEvents(observer)
  await drain(observer, [{ type: "finish", reason: { kind: "stop" } }])
  const call = records.find((record) => record.record_type === "call/start")
  assert.equal(call.provider_attempt_status, "llm_stream_entered")
})
