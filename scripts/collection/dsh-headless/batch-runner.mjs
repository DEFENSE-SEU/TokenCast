import { createInterface } from "node:readline"
import { isAbsolute } from "node:path"

import { importDshHostModule } from "./host.mjs"
import { prepareHeadlessRunner, runHeadlessJob } from "./runner.mjs"

const schemastery = await importDshHostModule("@deepseek-ai/schemastery")
const z = schemastery.default ?? schemastery

export const BATCH_PROTOCOL_VERSION = 2
export const name = "dsh-headless-batch-runner"
export const inject = ["agentPresets", "agents", "llm", "sessionPersistence", "sessions"]
export const Config = z.object({
  provider: z.string().required(),
  model: z.string().required(),
  agentPreset: z.string().required(),
  exclusiveProvider: z.string(),
})

function nonEmptyText(value, label) {
  if (typeof value !== "string" || value.length === 0) throw new Error(`${label} is required`)
  return value
}

function positiveInteger(value, label) {
  if (!Number.isInteger(value) || value < 1) throw new Error(`${label} must be a positive integer`)
  return value
}

function nullableCap(value, label) {
  return value === null ? null : positiveInteger(value, label)
}

export function normalizeJob(message) {
  const decoding = message.decoding
  const caps = message.caps
  if (decoding === null || typeof decoding !== "object") {
    throw new Error("job decoding is required")
  }
  if (caps === null || typeof caps !== "object") throw new Error("job caps are required")
  if (typeof decoding.temperature !== "number" || !Number.isFinite(decoding.temperature)) {
    throw new Error("job decoding.temperature must be finite")
  }
  if (!Array.isArray(decoding.stop) || decoding.stop.some((value) => typeof value !== "string")) {
    throw new Error("job decoding.stop must contain strings")
  }
  const retention = {}
  for (const key of ["tool_result_max_utf8_bytes", "tool_result_history_keep_recent"]) {
    if (message[key] === undefined) continue
    if (message[key] !== null && (!Number.isSafeInteger(message[key]) || message[key] < 0)) {
      throw new Error(`job ${key} must be null or a non-negative integer`)
    }
    retention[key] = message[key]
  }
  const promptPolicy = message.system_prompt_policy_id === undefined ? {}
    : { system_prompt_policy_id: nonEmptyText(message.system_prompt_policy_id, "job system_prompt_policy_id") }
  let budgetShadow = {}
  if (message.budget_shadow !== undefined) {
    const shadow = message.budget_shadow
    if (shadow === null || typeof shadow !== "object" || Array.isArray(shadow)) {
      throw new Error("job budget_shadow must be an object")
    }
    for (const key of ["output_path", "source_id", "task_family", "run_id"]) {
      nonEmptyText(shadow[key], `job budget_shadow.${key}`)
    }
    if (!isAbsolute(shadow.output_path)) {
      throw new Error("job budget_shadow.output_path must be absolute")
    }
    if (!Number.isSafeInteger(shadow.budget_tokens) || shadow.budget_tokens < 0
      || !Number.isSafeInteger(shadow.margin_tokens) || shadow.margin_tokens < 0) {
      throw new Error("job budget_shadow token bounds must be non-negative integers")
    }
    budgetShadow = { budget_shadow: {
      output_path: shadow.output_path,
      source_id: shadow.source_id,
      task_family: shadow.task_family,
      run_id: shadow.run_id,
      budget_tokens: shadow.budget_tokens,
      margin_tokens: shadow.margin_tokens,
    } }
  }
  return {
    ...retention,
    ...promptPolicy,
    ...budgetShadow,
    job_id: nonEmptyText(message.job_id, "job_id"),
    session_id: nonEmptyText(message.session_id, "session_id"),
    cwd: nonEmptyText(message.cwd, "cwd"),
    task: nonEmptyText(message.task, "task"),
    decoding: {
      reasoning_effort: nonEmptyText(decoding.reasoning_effort, "job decoding.reasoning_effort"),
      temperature: decoding.temperature,
      max_tokens: positiveInteger(decoding.max_tokens, "job decoding.max_tokens"),
      stop: [...decoding.stop],
    },
    caps: {
      max_logical_calls: nullableCap(caps.max_logical_calls, "job caps.max_logical_calls"),
      max_provider_accounted_tokens: nullableCap(
        caps.max_provider_accounted_tokens,
        "job caps.max_provider_accounted_tokens",
      ),
    },
  }
}

function errorText(error) {
  return error instanceof Error ? error.message : String(error)
}

export async function serveBatch(ctx, config, io) {
  await prepareHeadlessRunner(ctx, config)
  io.write({ type: "ready", protocol_version: BATCH_PROTOCOL_VERSION, pid: process.pid })
  for await (const line of io.lines) {
    let message
    try {
      message = JSON.parse(line)
    } catch (error) {
      io.write({
        type: "job/error",
        job_id: null,
        session_id: null,
        request_boundary_crossed: false,
        message: errorText(error),
      })
      continue
    }
    if (message?.type === "shutdown") break
    const jobId = typeof message?.job_id === "string" ? message.job_id : null
    const sessionId = typeof message?.session_id === "string" ? message.session_id : null
    let job
    try {
      if (message?.type !== "job") throw new Error("expected job or shutdown frame")
      job = normalizeJob(message)
    } catch (error) {
      io.write({
        type: "job/error",
        job_id: jobId,
        session_id: sessionId,
        request_boundary_crossed: false,
        message: errorText(error),
      })
      continue
    }
    let requestBoundaryCrossed = false
    try {
      const outcome = await runHeadlessJob(ctx, config, job, {
        started({ sessionId: startedSessionId, sessionPath }) {
          io.write({
            type: "job/started",
            job_id: job.job_id,
            session_id: startedSessionId,
            session_path: sessionPath,
          })
        },
        requestBoundaryCrossed() {
          requestBoundaryCrossed = true
        },
      })
      if (outcome.error !== undefined) {
        io.write({
          type: "job/error",
          job_id: job.job_id,
          session_id: outcome.sessionId,
          request_boundary_crossed: requestBoundaryCrossed,
          message: errorText(outcome.error),
        })
        continue
      }
      io.write({
        type: "job/done",
        job_id: job.job_id,
        session_id: outcome.sessionId,
        outcome: outcome.reason.kind,
        text: outcome.text,
      })
    } catch (error) {
      io.write({
        type: "fatal",
        job_id: job.job_id,
        session_id: job.session_id,
        request_boundary_crossed: requestBoundaryCrossed,
        message: errorText(error),
      })
      io.close()
      io.exit(1)
      return
    }
  }
  io.close()
  io.write({ type: "bye" })
  io.exit(0)
}

export function apply(ctx, config) {
  const exit = ctx.get("appExit")
  if (exit === undefined) {
    throw new Error("dsh-headless-batch-runner requires the DSH appExit service")
  }
  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity })
  const io = {
    lines,
    write(frame) {
      process.stdout.write(`${JSON.stringify(frame)}\n`)
    },
    close() {
      lines.close()
    },
    exit,
  }
  serveBatch(ctx, config, io).catch((error) => {
    lines.close()
    process.stderr.write(`dsh: ${errorText(error)}\n`)
    exit(1)
  })
}
