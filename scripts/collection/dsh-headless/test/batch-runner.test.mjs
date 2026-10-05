import assert from "node:assert/strict"
import { mkdtempSync, readFileSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { dirname, join } from "node:path"
import test from "node:test"
import { fileURLToPath } from "node:url"

let hostModules
try {
  const batchRunner = await import("../batch-runner.mjs")
  const fixtureAdapter = await import("../fixture-adapter.mjs")
  const { importDshHostModule } = await import("../host.mjs")
  const llm = await importDshHostModule("@deepseek-ai/dsh-llm")
  hostModules = { ...batchRunner, ...fixtureAdapter, ...llm }
} catch (error) {
  const missingHost = ["MODULE_NOT_FOUND", "ERR_MODULE_NOT_FOUND"].includes(error?.code)
    && String(error?.message).includes("@deepseek-ai/")
  if (!missingHost) throw error
}

const FIXTURE_RETRY_CODE = hostModules?.FIXTURE_RETRY_CODE
const markAgentLoopRequest = hostModules?.markAgentLoopRequest
const FixtureAdapter = hostModules?.FixtureAdapter
const serveBatch = hostModules?.serveBatch
const normalizeJob = hostModules?.normalizeJob

function hostBackedTest(name, body) {
  if (hostModules) return test(name, body)
  return test(name, { skip: "DSH host modules are unavailable" }, body)
}
const here = dirname(fileURLToPath(import.meta.url))

function job(jobId, sessionId, cwd, task, maxTokens = 64) {
  return {
    type: "job",
    job_id: jobId,
    session_id: sessionId,
    cwd,
    task,
    decoding: {
      reasoning_effort: "off",
      temperature: 0,
      max_tokens: maxTokens,
      stop: [],
    },
    caps: {
      max_logical_calls: 12,
      max_provider_accounted_tokens: 30_000,
    },
  }
}

hostBackedTest("batch retention settings preserve zero, null and recent-window policies", () => {
  const base = job("job", "session", "C:/work", "task")
  const normalized = normalizeJob({ ...base, tool_result_max_utf8_bytes: 0, tool_result_history_keep_recent: 2 })
  assert.equal(normalized.tool_result_max_utf8_bytes, 0)
  assert.equal(normalized.tool_result_history_keep_recent, 2)
  assert.equal(normalizeJob({ ...base, tool_result_max_utf8_bytes: null }).tool_result_max_utf8_bytes, null)
  assert.equal(Object.hasOwn(normalizeJob(base), "tool_result_history_keep_recent"), false)
  for (const value of [-1, 1.5, true, "2", Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(() => normalizeJob({ ...base, tool_result_history_keep_recent: value }), /non-negative integer/)
  }
})

hostBackedTest("batch preserves the versioned system prompt policy", () => {
  const base = job("job", "session", "C:/work", "task")
  assert.equal(
    normalizeJob({ ...base, system_prompt_policy_id: "swebench-source-patch-submit-v5" }).system_prompt_policy_id,
    "swebench-source-patch-submit-v5",
  )
  assert.equal(Object.hasOwn(normalizeJob(base), "system_prompt_policy_id"), false)
  assert.throws(() => normalizeJob({ ...base, system_prompt_policy_id: "" }), /system_prompt_policy_id/)
})

hostBackedTest("budget shadow requires an isolated source, family, and absolute output", () => {
  const base = job("job", "session", "C:/work", "task")
  const shadow = { output_path: "C:/capture/shadow.jsonl", source_id: "dev-source",
    task_family: "repo_repair", run_id: "r1", budget_tokens: 300, margin_tokens: 64 }
  assert.deepEqual(normalizeJob({ ...base, budget_shadow: shadow }).budget_shadow, shadow)
  assert.throws(() => normalizeJob({ ...base, budget_shadow: { ...shadow, source_id: "" } }), /source_id/)
  assert.throws(() => normalizeJob({ ...base, budget_shadow: { ...shadow, output_path: "relative.jsonl" } }), /absolute/)
})

async function* lines(frames) {
  for (const frame of frames) yield JSON.stringify(frame)
}

function fakeContext({
  failingSessions = [],
  flushFailures = [],
  disposalFailures = [],
  usageLessRetrySessions = [],
  postRequestFailures = [],
  shadowStreamSessions = [],
} = {}) {
  const failures = new Set(failingSessions)
  const failedFlushes = new Set(flushFailures)
  const failedDisposals = new Set(disposalFailures)
  const usageLessRetries = new Set(usageLessRetrySessions)
  const requestFailures = new Set(postRequestFailures)
  const shadowStreams = new Set(shadowStreamSessions)
  const records = {
    active: new Set(),
    creates: [],
    disposals: [],
    flushes: [],
    followups: [],
    loaderAwaits: 0,
    loads: [],
    locations: [],
    requests: [],
  }
  const ctx = {
    get(name) {
      if (name === "tokenMeter" && shadowStreams.size > 0) {
        return { measure: () => ({ totalTokens: 120, baseline: { kind: "estimated" } }) }
      }
      if (name !== "loader") return undefined
      return {
        async await() {
          records.loaderAwaits += 1
        },
      }
    },
    llm: {
      listProviders() {
        return [{ id: "dsh-fixture" }]
      },
      async resolveModelInfo(provider, model) {
        return { provider, id: model, name: model }
      },
    },
    agentPresets: {
      async mount(_agentCtx, preset) {
        return { id: preset }
      },
    },
    agents: {
      async create(options) {
        const sessionId = String(options.sessionId)
        const listeners = new Map()
        const agentCtx = {
          on(name, handler) {
            const registered = listeners.get(name) ?? []
            registered.push(handler)
            listeners.set(name, registered)
            return () => undefined
          },
          tools: {
            restrict() {},
          },
        }
        await options.setup(agentCtx)
        const session = {
          id: sessionId,
          header: {
            id: sessionId,
            cwd: options.meta.cwd,
            agentPreset: options.meta.agentPreset,
          },
          seq: 0,
          events: [],
          requestHeader() { return { config: { provider: "dsh-fixture", model: "dsh-fixture-model" } } },
        }
        let followup
        let completed = false
        const agent = {
          session,
          async whenIdle() {
            if (followup === undefined || completed) return
            if (failures.has(sessionId)) throw new Error(`fake job failure: ${sessionId}`)
            const requestHandlers = listeners.get("agent/request") ?? []
            const payload = { turn: 1, step: 1 }
            const request = await requestHandlers.at(-1)(payload, async () => ({}))
            records.requests.push({ sessionId, request })
            if (requestFailures.has(sessionId)) {
              throw new Error(`fake post-request failure: ${sessionId}`)
            }
            if (usageLessRetries.has(sessionId)) {
              session.events.push({
                type: "assistant/chunk",
                seq: session.seq++,
                data: {
                  turn: 1,
                  step: 1,
                  chunk: { type: "finish", reason: { kind: "error" } },
                },
              })
              const retry = await requestHandlers.at(-1)(payload, async () => ({}))
              records.requests.push({ sessionId, request: retry })
            }
            if (shadowStreams.has(sessionId)) {
              const handler = (listeners.get("llm/stream") ?? []).at(-1)
              assert.equal(typeof handler, "function")
              const stream = handler(markAgentLoopRequest({
                sessionId, provider: "dsh-fixture", model: "dsh-fixture-model",
              }), () => (async function* () {
                yield { type: "usage", usage: { inputTokens: 9, outputTokens: 2,
                  cacheReadTokens: 3 } }
                yield { type: "finish", reason: { kind: "completed" } }
              })())
              for await (const _chunk of stream) {   }
            }
            session.events.push(
              { type: "turn/start", seq: 0, data: {} },
              {
                type: "assistant/message",
                seq: 1,
                data: { message: { content: [{ type: "text", text: `done:${sessionId}` }] } },
              },
              { type: "turn/end", seq: 2, data: { reason: { kind: "completed" } } },
            )
            session.seq = 3
            completed = true
          },
          followup(message) {
            followup = message
            records.followups.push({ sessionId, message })
          },
        }
        const handle = {
          agent,
          async dispose() {
            records.disposals.push(sessionId)
            if (failedDisposals.has(sessionId)) {
              throw new Error(`fake disposal failure: ${sessionId}`)
            }
            records.active.delete(sessionId)
          },
        }
        records.active.add(sessionId)
        records.creates.push({ sessionId, options })
        return handle
      },
    },
    sessionPersistence: {
      locate(header) {
        records.locations.push(header)
        return {
          kind: "jsonl",
          path: `C:/sessions/${header.id}/session.jsonl.zstd`,
        }
      },
      async load(sessionId) {
        records.loads.push(String(sessionId))
        return { meta: { id: String(sessionId) }, events: [] }
      },
    },
    sessions: {
      async flush(session) {
        records.flushes.push(session.id)
        if (failedFlushes.has(session.id)) {
          throw new Error(`fake flush failure: ${session.id}`)
        }
        return true
      },
    },
  }
  return { ctx, records }
}

function fakeIo(frames, records) {
  const writes = []
  const exits = []
  const closes = []
  return {
    io: {
      lines: lines(frames),
      write(frame) {
        writes.push({ frame, disposedCount: records.disposals.length })
      },
      exit(code) {
        exits.push(code)
      },
      close() {
        closes.push(true)
      },
    },
    writes,
    exits,
    closes,
  }
}

const config = {
  provider: "dsh-fixture",
  model: "dsh-fixture-model",
  agentPreset: "router-standard",
  exclusiveProvider: "dsh-fixture",
}

test("bundle selects exactly one DSH runner from DSH_BATCH", () => {
  const patch = readFileSync(join(here, "../cordis.patch.yml"), "utf8")
  const packageJson = JSON.parse(readFileSync(join(here, "../package.json"), "utf8"))

  assert.match(
    patch,
    /id: dsh-headless-runner[\s\S]*?disabled: !!js process\.env\.DSH_BATCH === '1'/,
  )
  assert.match(
    patch,
    /id: dsh-batch-runner[\s\S]*?disabled: !!js process\.env\.DSH_BATCH !== '1'/,
  )
  assert.equal(packageJson.exports["./batch-runner"], "./batch-runner.mjs")
  assert.ok(packageJson.files.includes("batch-runner.mjs"))
})

hostBackedTest("batch runner isolates two jobs in one process and disposes each handle", async () => {
  const { ctx, records } = fakeContext()
  const first = job("job-1", "session-1", "C:/work/one", "first task", 64)
  const second = job("job-2", "session-2", "C:/work/two", "second task", 96)
  const { io, writes, exits, closes } = fakeIo(
    [first, second, { type: "shutdown" }],
    records,
  )

  await serveBatch(ctx, config, io)

  assert.deepEqual(writes.map(({ frame }) => frame.type), [
    "ready",
    "job/started",
    "job/done",
    "job/started",
    "job/done",
    "bye",
  ])
  assert.equal(writes[0].frame.pid, process.pid)
  assert.equal(writes[0].frame.protocol_version, 2)
  assert.deepEqual(
    records.creates.map(({ sessionId, options }) => ({
      sessionId,
      cwd: options.meta.cwd,
      maxTokens: options.agentOptions.maxTokens,
    })),
    [
      { sessionId: "session-1", cwd: "C:/work/one", maxTokens: 64 },
      { sessionId: "session-2", cwd: "C:/work/two", maxTokens: 96 },
    ],
  )
  assert.deepEqual(records.locations.map(({ id, cwd }) => ({ id, cwd })), [
    { id: "session-1", cwd: "C:/work/one" },
    { id: "session-2", cwd: "C:/work/two" },
  ])
  assert.deepEqual(records.followups.map(({ sessionId, message }) => ({
    sessionId,
    text: message.content[0].text,
  })), [
    { sessionId: "session-1", text: "first task" },
    { sessionId: "session-2", text: "second task" },
  ])
  assert.deepEqual(records.requests.map(({ sessionId, request }) => ({
    sessionId,
    reasoningEffort: request.reasoningEffort,
    temperature: request.temperature,
    maxTokens: request.maxTokens,
    stop: request.stop,
  })), [
    {
      sessionId: "session-1",
      reasoningEffort: "off",
      temperature: 0,
      maxTokens: 64,
      stop: [],
    },
    {
      sessionId: "session-2",
      reasoningEffort: "off",
      temperature: 0,
      maxTokens: 96,
      stop: [],
    },
  ])
  assert.deepEqual(records.flushes, ["session-1", "session-2"])
  assert.deepEqual(records.loads, ["session-1", "session-2"])
  assert.deepEqual(records.disposals, ["session-1", "session-2"])
  assert.equal(records.active.size, 0)
  assert.equal(records.loaderAwaits, 1)
  assert.equal(writes.find(({ frame }) => frame.job_id === "job-1" && frame.type === "job/done").disposedCount, 1)
  assert.equal(writes.find(({ frame }) => frame.job_id === "job-2" && frame.type === "job/done").disposedCount, 2)
  assert.equal(writes[1].frame.session_path, "C:/sessions/session-1/session.jsonl.zstd")
  assert.equal(writes[3].frame.session_path, "C:/sessions/session-2/session.jsonl.zstd")
  assert.deepEqual(closes, [true])
  assert.deepEqual(exits, [0])
})

hostBackedTest("batch opt-in budget shadow traces a fixture DSH stream", async (t) => {
  const directory = mkdtempSync(join(tmpdir(), "dsh-batch-shadow-"))
  t.after(() => rmSync(directory, { recursive: true, force: true }))
  const outputPath = join(directory, "shadow.jsonl")
  const { ctx, records } = fakeContext({ shadowStreamSessions: ["shadow-session"] })
  const frame = {
    ...job("shadow-job", "shadow-session", "C:/work/shadow", "exposed task"),
    budget_shadow: {
      output_path: outputPath, source_id: "dsh-exposed-dev",
      task_family: "repo_repair", run_id: "exposed-1",
      budget_tokens: 50, margin_tokens: 0,
    },
  }
  const { io, writes } = fakeIo([frame, { type: "shutdown" }], records)
  await serveBatch(ctx, config, io)
  assert.deepEqual(writes.map(({ frame: item }) => item.type),
    ["ready", "job/started", "job/done", "bye"])
  const trace = readFileSync(outputPath, "utf8").trim().split("\n").map(JSON.parse)
  assert.deepEqual(trace.map((row) => row.record_type),
    ["scope", "pre_dispatch", "attempt_end"])
  assert.equal(trace[0].task_family, "repo_repair")
  assert.equal(trace[1].dsh_meter_estimate_tokens, 120)
  assert.equal(trace[1].would_stop, true)
  assert.equal(trace[2].usage.provider_accounted_tokens, 14)
  assert.equal(trace[2].usage.cache_read_tokens, 3)
})

hostBackedTest("a usage-less retry remains inside one logical Call cap", async () => {
  const { ctx, records } = fakeContext({ usageLessRetrySessions: ["session-retry"] })
  const retryJob = job("job-retry", "session-retry", "C:/work/retry", "retry task")
  retryJob.caps.max_logical_calls = 1
  const { io, writes, exits } = fakeIo(
    [retryJob, { type: "shutdown" }],
    records,
  )

  await serveBatch(ctx, config, io)

  assert.deepEqual(writes.map(({ frame }) => frame.type), [
    "ready",
    "job/started",
    "job/done",
    "bye",
  ])
  assert.equal(records.requests.length, 2)
  assert.deepEqual(records.requests.map(({ sessionId }) => sessionId), [
    "session-retry",
    "session-retry",
  ])
  assert.deepEqual(exits, [0])
})

hostBackedTest("batch runner disposes a failed job and accepts the next job", async () => {
  const { ctx, records } = fakeContext({ failingSessions: ["session-fail"] })
  const failing = job("job-fail", "session-fail", "C:/work/fail", "fail task")
  const succeeding = job("job-ok", "session-ok", "C:/work/ok", "ok task")
  const { io, writes, exits, closes } = fakeIo(
    [failing, succeeding, { type: "shutdown" }],
    records,
  )

  await serveBatch(ctx, config, io)

  assert.deepEqual(writes.map(({ frame }) => frame.type), [
    "ready",
    "job/started",
    "job/error",
    "job/started",
    "job/done",
    "bye",
  ])
  const failed = writes.find(({ frame }) => frame.type === "job/error")
  assert.equal(failed.frame.job_id, "job-fail")
  assert.equal(failed.frame.session_id, "session-fail")
  assert.equal(failed.frame.request_boundary_crossed, false)
  assert.match(failed.frame.message, /fake job failure/)
  assert.equal(failed.disposedCount, 1)
  assert.equal(writes.find(({ frame }) => frame.job_id === "job-ok" && frame.type === "job/done").disposedCount, 2)
  assert.deepEqual(records.flushes, ["session-fail", "session-ok"])
  assert.deepEqual(records.loads, ["session-fail", "session-ok"])
  assert.deepEqual(records.disposals, ["session-fail", "session-ok"])
  assert.equal(records.active.size, 0)
  assert.deepEqual(closes, [true])
  assert.deepEqual(exits, [0])
})

hostBackedTest("job errors retain the crossed Request boundary", async () => {
  const { ctx, records } = fakeContext({ postRequestFailures: ["session-fail"] })
  const failing = job("job-fail", "session-fail", "C:/work/fail", "fail task")
  const { io, writes, exits } = fakeIo([failing, { type: "shutdown" }], records)

  await serveBatch(ctx, config, io)

  const failed = writes.find(({ frame }) => frame.type === "job/error")
  assert.equal(failed.frame.request_boundary_crossed, true)
  assert.match(failed.frame.message, /post-request failure/)
  assert.deepEqual(exits, [0])
})

hostBackedTest("batch runner stops after session cleanup fails", async (context) => {
  for (const failure of ["flush", "dispose"]) {
    await context.test(failure, async () => {
      const options = failure === "flush"
        ? { flushFailures: ["session-fail"] }
        : { disposalFailures: ["session-fail"] }
      const { ctx, records } = fakeContext(options)
      const first = job("job-fail", "session-fail", "C:/work/fail", "first task")
      const second = job("job-unused", "session-unused", "C:/work/unused", "second task")
      const { io, writes, exits, closes } = fakeIo(
        [first, second, { type: "shutdown" }],
        records,
      )

      await serveBatch(ctx, config, io)

      assert.deepEqual(writes.map(({ frame }) => frame.type), [
        "ready",
        "job/started",
        "fatal",
      ])
      assert.equal(writes[2].frame.job_id, "job-fail")
      assert.equal(writes[2].frame.session_id, "session-fail")
      assert.equal(writes[2].frame.request_boundary_crossed, true)
      const failureLabel = failure === "dispose" ? "disposal" : failure
      assert.match(writes[2].frame.message, new RegExp(`fake ${failureLabel} failure`))
      assert.deepEqual(records.creates.map(({ sessionId }) => sessionId), ["session-fail"])
      assert.deepEqual(closes, [true])
      assert.deepEqual(exits, [1])
    })
  }
})

function fixtureRequest(sessionId) {
  return markAgentLoopRequest({
    sessionId,
    provider: "dsh-fixture",
    model: "dsh-fixture-model",
    reasoningEffort: "off",
    temperature: 0,
    maxTokens: 64,
    stop: [],
  })
}

async function fixtureAttempt(adapter, sessionId) {
  const events = []
  try {
    for await (const event of adapter.stream(fixtureRequest(sessionId))) events.push(event)
    return { events, error: undefined }
  } catch (error) {
    return { events, error }
  }
}

hostBackedTest("fixture retry counters are independent across fresh sessions", async () => {
  const adapter = new FixtureAdapter()

  const firstA = await fixtureAttempt(adapter, "session-a")
  const firstB = await fixtureAttempt(adapter, "session-b")
  const secondA = await fixtureAttempt(adapter, "session-a")
  const secondB = await fixtureAttempt(adapter, "session-b")
  const thirdA = await fixtureAttempt(adapter, "session-a")

  assert.equal(firstA.error.code, FIXTURE_RETRY_CODE)
  assert.equal(firstB.error.code, FIXTURE_RETRY_CODE)
  assert.equal(secondA.error, undefined)
  assert.equal(secondB.error, undefined)
  assert.equal(secondA.events.at(-1).type, "finish")
  assert.equal(secondB.events.at(-1).type, "finish")
  assert.equal(thirdA.error.code, "DSH_FIXTURE_OVERFLOW")
})


hostBackedTest("explicit null call and token budgets allow normal job execution", async () => {
  const { ctx, records } = fakeContext()
  const unlimited = job("unlimited", "session-unlimited", "C:/work/unlimited", "complete task")
  unlimited.caps.max_logical_calls = null
  unlimited.caps.max_provider_accounted_tokens = null
  assert.deepEqual(normalizeJob(unlimited).caps, {
    max_logical_calls: null, max_provider_accounted_tokens: null,
  })
  const { io, writes, exits } = fakeIo([unlimited, { type: "shutdown" }], records)
  await serveBatch(ctx, config, io)
  assert.equal(writes.some(({ frame }) => frame.type === "job/done"), true)
  assert.ok(records.requests.length > 0)
  assert.deepEqual(exits, [0])
})


hostBackedTest("single-run schema preserves disabled cumulative budgets", async () => {
  const { Config } = await import("../runner.mjs")
  const normalized = Config({task: "offline", provider: "fixture", model: "fixture",
    agentPreset: "router-standard", reasoningEffort: "off", temperature: 0, maxTokens: 8192,
    maxLogicalCalls: undefined, maxProviderAccountedTokens: undefined})
  assert.equal(normalized.maxLogicalCalls, undefined)
  assert.equal(normalized.maxProviderAccountedTokens, undefined)
})
