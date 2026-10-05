import { createHash } from "node:crypto"
import { existsSync, readFileSync } from "node:fs"
import { dirname, isAbsolute, join, resolve } from "node:path"
import { pathToFileURL } from "node:url"

import { createPaidBudgetGate } from "./paid-budget-gate.mjs"

function readCompleteJsonl(path) {
  const source = readFileSync(path)
  const text = source.toString("utf8")
  if (!text.endsWith("\n")) throw new Error("Paid budget: incomplete observer JSONL")
  return { sha256: createHash("sha256").update(source).digest("hex"),
    rows: text.trimEnd().split("\n").map((line) => JSON.parse(line)) }
}

function capturePairs(capturePath) {
  const capture = JSON.parse(readFileSync(capturePath, "utf8"))
  if (capture.condition?.configured_model !== "openai/gpt-5.4"
    || capture.condition?.configured_provider !== "qwen-compatible"
    || capture.condition?.reasoning_effort !== "low"
    || capture.runtime?.hosted_provider?.base_url !== "https://openai.sufy.com/v1"
    || capture.runtime?.max_attempts_per_logical_call !== 1
    || capture.runtime?.opaque_provider_retry_allowed !== false
    || capture.terminal?.attempt_ledger_complete !== true
    || typeof capture.capture_id !== "string" || !capture.capture_id
    || typeof capture.session_id !== "string" || !capture.session_id) {
    throw new Error("Paid budget: capture route or terminal contract drift")
  }
  const { sha256, rows } = readCompleteJsonl(join(dirname(capturePath), "observer.jsonl"))
  if (rows[0]?.record_type !== "observer/session"
    || rows[0]?.session_id !== capture.session_id) {
    throw new Error("Paid budget: observer session mismatch")
  }
  const pairs = []
  let active = null
  let priorEnd = -Infinity
  const ids = new Set()
  for (const row of rows.slice(1)) {
    if (row.session_id !== capture.session_id) {
      throw new Error("Paid budget: mixed observer sessions")
    }
    if (row.record_type === "attempt/start") {
      if (active || ids.has(row.attempt_id) || row.attempt_ordinal !== 1
        || row.provider !== "qwen-compatible" || row.model !== "openai/gpt-5.4"
        || row.reasoning_effort !== "low" || !Number.isSafeInteger(row.time)
        || row.time < priorEnd) {
        throw new Error("Paid budget: overlapping or invalid observer attempt")
      }
      ids.add(row.attempt_id)
      active = row
    } else if (row.record_type === "attempt/end") {
      if (!active || row.attempt_id !== active.attempt_id
        || row.logical_call_id !== active.logical_call_id
        || !Number.isSafeInteger(row.time) || row.time < active.time) {
        throw new Error("Paid budget: invalid observer terminal")
      }
      pairs.push({ start: active, end: row })
      priorEnd = row.time
      active = null
    } else if (row.record_type === "attempt/checkpoint") {
      if (!active || row.attempt_id !== active.attempt_id) {
        throw new Error("Paid budget: orphan observer checkpoint")
      }
    } else throw new Error("Paid budget: unexpected observer record")
  }
  if (active || !pairs.length) throw new Error("Paid budget: incomplete observer attempts")
  return { capture, observerSha256: sha256, pairs }
}

export function settlePaidBudgetCapture(ledgerPath, capturePath) {
  if (!isAbsolute(ledgerPath) || !isAbsolute(capturePath) || !existsSync(ledgerPath)) {
    throw new Error("Paid budget: existing absolute ledger and capture paths required")
  }
  const gate = createPaidBudgetGate(ledgerPath)
  try {
    const { capture, observerSha256, pairs } = capturePairs(capturePath)
    const attempts = gate.snapshot()
    const matched = new Set()
    const ready = []
    let retained = 0
    let lastLedgerAttempt = 0
    for (const { start, end } of pairs) {
      const candidates = attempts.filter((item) => {
        const time = Date.parse(item.reserve.at)
        return Number.isFinite(time) && time >= start.time && time <= end.time
      })
      if (candidates.length > 1) throw new Error("Paid budget: ambiguous ledger attempt interval")
      if (candidates.length === 0) {
        const usage = end.usage
        if (end.failure_code == null || end.observer_finalized !== true
          || usage && (usage.inputTokens !== 0 || usage.outputTokens !== 0)) {
          throw new Error("Paid budget: unreserved observer attempt is not a pre-send failure")
        }
        continue
      }
      const item = candidates[0]
      if (matched.has(item.attempt) || item.attempt <= lastLedgerAttempt) {
        throw new Error("Paid budget: duplicate or unordered ledger match")
      }
      lastLedgerAttempt = item.attempt
      matched.add(item.attempt)
      if (item.http !== 200 && end.failure_code === null && end.observer_finalized === true) {
        throw new Error("Paid budget: successful observer attempt lacks HTTP 200")
      }
      if (item.http !== 200 || !item.sse || end.failure_code !== null
        || end.observer_finalized !== true) {
        retained += 1
        continue
      }
      const usage = end.usage
      if (!usage || !Number.isSafeInteger(usage.inputTokens)
        || !Number.isSafeInteger(usage.outputTokens)
        || usage.cacheReadTokens != null || usage.cacheWriteTokens != null
        || usage.inputTokens !== item.sse.usage.prompt_tokens
        || usage.outputTokens !== item.sse.usage.completion_tokens) {
        throw new Error("Paid budget: capture and raw SSE usage disagree")
      }
      const evidence = { captureId: capture.capture_id,
        observerAttemptId: end.attempt_id, observerSha256,
        inputTokens: usage.inputTokens, outputTokens: usage.outputTokens }
      if (item.settled) {
        if (item.settled.capture_id !== evidence.captureId
          || item.settled.observer_attempt_id !== evidence.observerAttemptId
          || item.settled.observer_sha256 !== evidence.observerSha256) {
          throw new Error("Paid budget: conflicting existing settlement")
        }
      } else ready.push({ attempt: item.attempt, evidence })
    }
    const first = pairs[0].start.time
    const last = pairs.at(-1).end.time
    if (attempts.some((item) => {
      const time = Date.parse(item.reserve.at)
      return time >= first && time <= last && !matched.has(item.attempt)
    })) throw new Error("Paid budget: unmatched reservation in capture interval")
    let releasedUnits = 0
    for (const { attempt, evidence } of ready) {
      releasedUnits += gate.settle(attempt, evidence).released_units
    }
    return { captureId: capture.capture_id, settled: ready.length,
      alreadySettled: matched.size - ready.length - retained,
      retained, releasedYuan: releasedUnits / 10_000 }
  } finally {
    gate.close()
  }
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  try {
    if (process.argv.length !== 4) throw new Error("Usage: node settle-paid-budget.mjs <absolute-ledger> <absolute-capture.json>")
    process.stdout.write(`${JSON.stringify(settlePaidBudgetCapture(process.argv[2], process.argv[3]))}\n`)
  } catch (error) {
    process.stderr.write(`${error.message}\n`)
    process.exitCode = 1
  }
}
