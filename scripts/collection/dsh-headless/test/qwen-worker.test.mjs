import assert from "node:assert/strict"
import {
  mkdtempSync,
  readFileSync,
  rmSync,
} from "node:fs"
import { tmpdir } from "node:os"
import test from "node:test"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

let adapterModule
try {
  adapterModule = await import("../qwen-adapter.mjs")
} catch (error) {
  const missingHost = ["MODULE_NOT_FOUND", "ERR_MODULE_NOT_FOUND"].includes(error?.code)
    && String(error?.message).includes("@deepseek-ai/")
  if (!missingHost) throw error
}

const LOCAL_QWEN_MODEL = adapterModule?.LOCAL_QWEN_MODEL
  ?? "Qwen/Qwen2.5-7B-Instruct"
const LOCAL_QWEN_PROVIDER = adapterModule?.LOCAL_QWEN_PROVIDER ?? "local-qwen"
const LocalQwenAdapter = adapterModule?.LocalQwenAdapter

function hostBackedTest(name, body) {
  if (adapterModule) return test(name, body)
  return test(name, { skip: "DSH host modules are unavailable" }, body)
}

const here = dirname(fileURLToPath(import.meta.url))
const fakeWorker = join(here, "fake-qwen-worker.mjs")

function adapterOptions(overrides = {}) {
  return {
    provider: LOCAL_QWEN_PROVIDER,
    model: LOCAL_QWEN_MODEL,
    reasoningEffort: "off",
    messages: [
      {
        role: "user",
        content: [{ type: "text", text: "reply" }],
        source: { kind: "user" },
      },
    ],
    tools: [],
    temperature: 0,
    maxTokens: 8,
    stop: [],
    ...overrides,
  }
}

function createAdapter(logPath, config = {}) {
  return new LocalQwenAdapter({
    python: process.execPath,
    worker: fakeWorker,
    snapshot: logPath,
    ...config,
  })
}

function readLog(path) {
  try {
    return readFileSync(path, "utf8")
      .trim()
      .split(/\r?\n/)
      .filter(Boolean)
      .map((line) => JSON.parse(line))
  } catch (error) {
    if (error.code === "ENOENT") return []
    throw error
  }
}

function pidAlive(pid) {
  try {
    process.kill(pid, 0)
    return true
  } catch (error) {
    if (error.code === "ESRCH") return false
    throw error
  }
}

async function waitUntil(predicate, timeoutMs = 2_000) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (predicate()) return true
    await new Promise((resolve) => setTimeout(resolve, 20))
  }
  return predicate()
}

async function settleWithin(promise, timeoutMs = 2_000) {
  let timer
  const timeout = new Promise((resolve) => {
    timer = setTimeout(() => resolve({ status: "timeout" }), timeoutMs)
  })
  const settled = promise.then(
    (value) => ({ status: "fulfilled", value }),
    (error) => ({ status: "rejected", error }),
  )
  const outcome = await Promise.race([settled, timeout])
  clearTimeout(timer)
  return outcome
}

hostBackedTest("two completed Requests reuse one Qwen worker process", async () => {
  const directory = mkdtempSync(join(tmpdir(), "qwen-worker-reuse-"))
  const logPath = join(directory, "worker.log")
  const probeDir = join(directory, "probe")
  const adapter = createAdapter(logPath, {
    generationProbeDir: probeDir,
    generationProbeCaptureRevisionPrefix: "probe-capture-v3:run-001",
  })
  try {
    const firstChunks = []
    for await (const chunk of adapter.stream(
      adapterOptions({ sessionId: "session-a" }),
    )) firstChunks.push(chunk)
    const firstLog = readLog(logPath)
    const firstSpawn = firstLog.find((event) => event.type === "spawn")
    assert.equal(firstLog.filter((event) => event.type === "spawn").length, 1)
    assert.equal(firstLog.filter((event) => event.type === "request").length, 1)
    assert.equal(pidAlive(firstSpawn.pid), true)

    const secondChunks = []
    for await (const chunk of adapter.stream(adapterOptions({
      sessionId: "session-b",
      messages: [
        {
          role: "user",
          content: [{ type: "text", text: "second reply" }],
          source: { kind: "user" },
        },
      ],
    }))) secondChunks.push(chunk)

    const log = readLog(logPath)
    const spawns = log.filter((event) => event.type === "spawn")
    const requests = log.filter((event) => event.type === "request")
    const acknowledgements = log.filter((event) => event.type === "ack")
    assert.equal(spawns.length, 1)
    assert.equal(requests.length, 2)
    assert.equal(acknowledgements.length, 2)
    assert.equal(requests[0].pid, spawns[0].pid)
    assert.equal(requests[1].pid, spawns[0].pid)
    assert.notEqual(requests[0].id, requests[1].id)
    assert.deepEqual(
      requests.map((request) => request.generationProbe),
      [
        {
          path: join(probeDir, "session-a.jsonl"),
          capture_revision: "probe-capture-v3:run-001:session-a",
          session_id: "session-a",
          model_id: LOCAL_QWEN_MODEL,
        },
        {
          path: join(probeDir, "session-b.jsonl"),
          capture_revision: "probe-capture-v3:run-001:session-b",
          session_id: "session-b",
          model_id: LOCAL_QWEN_MODEL,
        },
      ],
    )
    assert.equal(pidAlive(spawns[0].pid), true)
    assert.equal(firstChunks.find((chunk) => chunk.type === "block-end").block.text, "reply")
    assert.equal(
      secondChunks.find((chunk) => chunk.type === "block-end").block.text,
      "second reply",
    )
  } finally {
    await adapter.dispose()
    rmSync(directory, { recursive: true, force: true })
  }
})

hostBackedTest("consumer return terminates the old worker and the next Request cold-starts", async () => {
  const directory = mkdtempSync(join(tmpdir(), "qwen-worker-return-"))
  const logPath = join(directory, "worker.log")
  const adapter = createAdapter(logPath)
  let secondIterator
  try {
    const firstIterator = adapter.stream(adapterOptions())
    assert.equal((await firstIterator.next()).value.type, "block-start")
    assert.equal((await firstIterator.next()).value.type, "text-delta")
    const firstPid = readLog(logPath).find((event) => event.type === "spawn").pid

    await firstIterator.return()
    const oldWorkerExited = await waitUntil(() => !pidAlive(firstPid))

    secondIterator = adapter.stream(adapterOptions())[Symbol.asyncIterator]()
    const secondChunks = []
    const secondCompletion = (async () => {
      for await (const chunk of secondIterator) secondChunks.push(chunk)
      return secondChunks
    })()
    const secondOutcome = await settleWithin(secondCompletion)

    assert.equal(oldWorkerExited, true, "the worker serving the abandoned stream stayed alive")
    assert.equal(secondOutcome.status, "fulfilled", "the next Request did not complete")
    assert.deepEqual(
      secondOutcome.value.map((chunk) => chunk.type),
      ["block-start", "text-delta", "block-end", "usage", "finish"],
    )
    const spawns = readLog(logPath).filter((event) => event.type === "spawn")
    assert.equal(spawns.length, 2)
    assert.notEqual(spawns[0].pid, spawns[1].pid)
  } finally {
    await adapter.dispose()
    await secondIterator?.return()
    rmSync(directory, { recursive: true, force: true })
  }
})

hostBackedTest("abort during ready loading terminates the worker promptly", async () => {
  const directory = mkdtempSync(join(tmpdir(), "qwen-worker-ready-abort-"))
  const logPath = join(directory, "ready-delay-worker.log")
  const adapter = createAdapter(logPath)
  const controller = new AbortController()
  const iterator = adapter.stream(adapterOptions({ signal: controller.signal }))
  const pending = iterator.next()
  try {
    assert.equal(await waitUntil(() => readLog(logPath).some((event) => event.type === "spawn")), true)
    const pid = readLog(logPath).find((event) => event.type === "spawn").pid
    controller.abort(new Error("cancel Qwen during ready"))

    const outcome = await settleWithin(pending)
    const workerExited = await waitUntil(() => !pidAlive(pid))

    assert.equal(outcome.status, "rejected", "ready-stage abort did not reject promptly")
    assert.equal(outcome.error, controller.signal.reason)
    assert.equal(workerExited, true, "ready-stage abort left the worker alive")
  } finally {
    await adapter.dispose()
    await pending.catch(() => undefined)
    await iterator.return()
    rmSync(directory, { recursive: true, force: true })
  }
})

hostBackedTest("worker transport preserves UTF-8 messages and fixes Python stream encoding", async () => {
  const directory = mkdtempSync(join(tmpdir(), "qwen-worker-utf8-"))
  const logPath = join(directory, "worker.log")
  const adapter = createAdapter(logPath)
  try {
    const chunks = []
    for await (const chunk of adapter.stream(
      adapterOptions({
        messages: [
          {
            role: "user",
            content: [{ type: "text", text: "\u4e2d\u6587\u8f68\u8ff9🚀" }],
            source: { kind: "user" },
          },
        ],
      }),
    )) {
      chunks.push(chunk)
    }
    const text = chunks.find((chunk) => chunk.type === "block-end").block.text
    const spawn = readLog(logPath).find((event) => event.type === "spawn")
    const request = readLog(logPath).find((event) => event.type === "request")

    assert.equal(text, "\u4e2d\u6587\u8f68\u8ff9🚀")
    assert.equal(request.output, "\u4e2d\u6587\u8f68\u8ff9🚀")
    assert.equal(spawn.pythonUtf8, "1")
    assert.equal(spawn.pythonIoEncoding, "utf-8")
    const workerSource = readFileSync(join(here, "../qwen-worker.py"), "utf8")
    assert.match(workerSource, /sys\.stdin\.reconfigure\(encoding="utf-8", errors="strict"\)/)
    assert.match(workerSource, /sys\.stdout\.reconfigure\(encoding="utf-8", errors="strict"\)/)
  } finally {
    await adapter.dispose()
    rmSync(directory, { recursive: true, force: true })
  }
})
