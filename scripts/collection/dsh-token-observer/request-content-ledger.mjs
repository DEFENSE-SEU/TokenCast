import { createHash } from "node:crypto"

const SOURCES = Symbol.for("token-prediction.request-content-sources.v1")
const POLICIES = Symbol.for("token-prediction.request-content-policies.v1")
const HISTORY_POLICIES = Symbol.for("token-prediction.request-history-policies.v1")
const sources = () => (globalThis[SOURCES] ??= new Map())
const policies = () => (globalThis[POLICIES] ??= new Map())
const historyPolicies = () => (globalThis[HISTORY_POLICIES] ??= new Map())
const hash = (text) => `sha256:${createHash("sha256").update(text, "utf8").digest("hex")}`
const escapePath = (value) => String(value).replaceAll("~", "~0").replaceAll("/", "~1")

export function canonicalRequestJson(value) {
  return JSON.stringify(value, (_key, item) => {
    if (item && typeof item === "object" && !Array.isArray(item)) {
      return Object.fromEntries(Object.keys(item).sort().map((key) => [key, item[key]]))
    }
    return item
  })
}

function requireJsonValue(value, seen = new Set()) {
  if (value === null || typeof value === "string" || typeof value === "boolean") return
  if (typeof value === "number" && Number.isFinite(value)) return
  if (typeof value !== "object" || seen.has(value)
    || (!Array.isArray(value) && Object.getPrototypeOf(value) !== Object.prototype)) {
    throw new TypeError("request value requires lossless JSON")
  }
  seen.add(value)
  for (const item of Object.values(value)) requireJsonValue(item, seen)
  seen.delete(value)
}

function sourceKey(sessionId, callId) {
  return JSON.stringify([String(sessionId), String(callId)])
}

export function disposeRequestContentSources(sessionId) {
  policies().delete(String(sessionId))
  historyPolicies().delete(String(sessionId))
  for (const key of sources().keys()) {
    if (JSON.parse(key)[0] === String(sessionId)) sources().delete(key)
  }
}

export function registerRequestContentPolicy(sessionId, maxUtf8Bytes) {
  retainToolResultContent([], maxUtf8Bytes)
  policies().set(String(sessionId), maxUtf8Bytes)
}

export function registerRequestHistoryPolicy(sessionId, keepRecent) {
  if (!Number.isSafeInteger(keepRecent) || keepRecent < 0) throw new TypeError("tool result history window must be a non-negative integer")
  historyPolicies().set(String(sessionId), keepRecent)
}

export function pruneToolResultHistory(session, { keepRecent, freezeMessage, estimateMessage }) {
  registerRequestHistoryPolicy(session.id, keepRecent)
  if (typeof freezeMessage !== "function" || typeof estimateMessage !== "function") {
    throw new TypeError("history pruning requires the DSH message freezer and token meter")
  }
  const candidates = [...session.surface.nodes].map((seq) => session.events[seq])
    .filter((event) => event.type === "tool/result")
  const expired = keepRecent === 0 ? candidates : candidates.slice(0, -keepRecent)
  const replacements = []
  for (const event of expired) {
    const message = event.data.message
    const result = message.content[0]
    if (result.content.length === 0) continue
    const key = sourceKey(session.id, result.toolCallId)
    const previous = sources().get(key)
    const replacementMessage = freezeMessage({ ...message, content: [{ ...result, content: [] }] })
    const shadowedTokenCount = estimateMessage(message)
    if (!Number.isFinite(shadowedTokenCount) || shadowedTokenCount < 0) {
      throw new TypeError("DSH token meter returned an invalid message estimate")
    }
    session.append("compaction/prune", {
      shadowedRange: { start: event.seq, end: event.seq }, shadowedSeqs: [event.seq], shadowedTokenCount,
    })
    const replacement = session.append("tool/result", { ...event.data, message: replacementMessage }, {
      surfaceOp: { op: "replace", start: event.seq, end: event.seq }, sourceEventSeqs: [event.seq],
    })
    sources().set(key, {
      ...previous, source: previous?.source ?? structuredClone(result.content), retained: [],
      historyOriginalSeq: event.seq, historyReplacementSeq: replacement.seq,
    })
    replacements.push({ original_seq: event.seq, replacement_seq: replacement.seq, tool_call_id: result.toolCallId })
  }
  return replacements
}

function utf8Prefix(text, limit) {
  let bytes = 0
  let result = ""
  for (const character of text.toWellFormed()) {
    const length = Buffer.byteLength(character, "utf8")
    if (bytes + length > limit) break
    result += character
    bytes += length
  }
  return result
}

export function retainToolResultContent(content, maxUtf8Bytes) {
  if (!Number.isSafeInteger(maxUtf8Bytes) || maxUtf8Bytes < 0) {
    throw new TypeError("tool result UTF-8 byte cap must be a non-negative integer")
  }
  let remaining = maxUtf8Bytes
  return content.map((block) => {
    if (block.type !== "text" && block.type !== "reasoning") {
      throw new TypeError(`tool result byte retention requires text/reasoning blocks: ${block.type}`)
    }
    if (typeof block.text !== "string") throw new TypeError("tool result text is missing")
    const text = utf8Prefix(block.text, remaining)
    remaining -= Buffer.byteLength(text, "utf8")
    // Stop the whole prefix when a character cannot fit, including later blocks.
    if (text.length < block.text.toWellFormed().length) remaining = 0
    return { ...block, text }
  })
}

export function createToolResultContentPolicy({ maxUtf8Bytes }) {
  retainToolResultContent([], maxUtf8Bytes)
  return async (exec, result, next) => {
    const decision = await next()
    if (exec.parent !== undefined) return decision
    if (Object.hasOwn(decision, "value")) {
      throw new TypeError("tool result retention cannot audit a post-execute value replacement")
    }
    const field = decision.kind === "block" ? "feedback" : "content"
    const original = decision[field] ?? result.content
    const retained = retainToolResultContent(original, maxUtf8Bytes)
    const sessionId = exec.agent?.session?.id
    if (!sessionId || !exec.callId) throw new TypeError("tool result retention requires session and call IDs")
    registerRequestContentPolicy(sessionId, maxUtf8Bytes)
    sources().set(sourceKey(sessionId, exec.callId), {
      source: structuredClone(original), retained: structuredClone(retained), maxUtf8Bytes,
    })
    return { ...decision, [field]: retained }
  }
}

function stringLeaves(value, path, visit) {
  if (typeof value === "string") visit(path, value)
  else if (Array.isArray(value)) value.forEach((item, index) => stringLeaves(item, `${path}/${index}`, visit))
  else if (value && typeof value === "object") {
    for (const [key, item] of Object.entries(value)) stringLeaves(item, `${path}/${escapePath(key)}`, visit)
  }
}

function atPath(value, path) {
  return path.split("/").slice(1).reduce((item, key) => item?.[key.replaceAll("~1", "/").replaceAll("~0", "~")], value)
}

export class RequestContentLedger {
  constructor({ writeRecord, now = Date.now } = {}) {
    if (typeof writeRecord !== "function") throw new TypeError("writeRecord must be a function")
    this.writeRecord = writeRecord
    this.now = now
    this.sessions = new Map()
  }

  stateFor(sessionId) {
    if (!this.sessions.has(sessionId)) this.sessions.set(sessionId, { nativeSeq: null, snapshots: new Map() })
    return this.sessions.get(sessionId)
  }

  observeNativeEvent(sessionId, event) {
    const state = this.stateFor(String(sessionId))
    if (Number.isSafeInteger(event.seq) && event.seq >= 0) state.nativeSeq = Math.max(state.nativeSeq ?? -1, event.seq)
  }

  disposeSession(sessionId) {
    this.sessions.delete(String(sessionId))
    disposeRequestContentSources(sessionId)
  }

  capture(request, facts) {
    const sessionId = String(facts.sessionId)
    const state = this.stateFor(sessionId)
    const missing = []
    const parts = []
    const removedParts = []
    const retention = []
    const snapshot = {}
    for (const [key, value] of Object.entries(request)) {
      if (key === "signal" || key === "sessionId") continue
      if (value === undefined) continue
      try {
        requireJsonValue(value)
        const serialized = canonicalRequestJson(value)
        if (serialized === undefined) throw new TypeError("unsupported JSON value")
        snapshot[key] = JSON.parse(serialized)
      } catch { missing.push(`request/${key}:json-unavailable`) }
    }
    if (!Array.isArray(snapshot.messages)) missing.push("request/messages")
    for (const field of ["provider", "model"]) {
      if (typeof snapshot[field] !== "string" || snapshot[field].length === 0) missing.push(`request/${field}`)
    }
    if (state.nativeSeq === null) missing.push("cutoff_native_seq")
    const add = (path, text, originId, sourceText = text, destination = parts) => {
      if (originId === null) missing.push(`${path}:origin_id`)
      const normalized = sourceText.toWellFormed()
      const retained = text.toWellFormed()
      if (!normalized.startsWith(retained)) {
        missing.push(`${path}:source-range-mismatch`)
        return
      }
      const sourceBytes = Buffer.byteLength(normalized, "utf8")
      const retainedBytes = Buffer.byteLength(retained, "utf8")
      // Include the origin so identical text from separate events stays distinguishable.
      destination.push({
        content_id: originId === null ? null : hash(canonicalRequestJson([originId, hash(normalized)])),
        origin_id: originId, path, source_text: normalized, source_utf8_bytes: sourceBytes,
        retained_ranges: retainedBytes === 0 ? [] : [[0, retainedBytes]], retained_utf8_bytes: retainedBytes,
      })
    }
    if (typeof snapshot.system === "string") add("/system", snapshot.system, `${sessionId}:system`)
    else if (snapshot.system !== undefined) missing.push("request/system:unsupported-shape")
    if (Array.isArray(snapshot.tools)) snapshot.tools.forEach((tool, index) => {
      add(`/tools/${index}`, canonicalRequestJson(tool),
        typeof tool?.name === "string" ? `${sessionId}:tool-schema:${tool.name}` : null)
    })
    else if (snapshot.tools !== undefined) missing.push("request/tools:unsupported-shape")
    for (const [index, message] of (Array.isArray(snapshot.messages) ? snapshot.messages : []).entries()) {
      if (message === null || typeof message !== "object") { missing.push(`/messages/${index}`); continue }
      if (!Array.isArray(message.content)) { missing.push(`/messages/${index}/content`); continue }
      message.content.forEach((block, blockIndex) => {
        const basePath = `/messages/${index}/content/${blockIndex}`
        if (block === null || typeof block !== "object") { missing.push(basePath); return }
        const callId = block.type === "tool-result" ? block.toolCallId : null
        if (block.type === "tool-result" && (typeof callId !== "string" || callId.length === 0)) {
          missing.push(`${basePath}:tool-call-id`)
        }
        const origin = callId ? `${sessionId}:tool-result:${callId}`
          : typeof message.id === "string" ? `${sessionId}:message:${message.id}:block:${blockIndex}` : null
        const provenance = callId ? sources().get(sourceKey(sessionId, callId)) : null
        if (callId && policies().has(sessionId) && !provenance) missing.push(`${basePath}:retention-provenance-missing`)
        if (provenance) retention.push({
          tool_call_id: callId,
          max_utf8_bytes: provenance.maxUtf8Bytes ?? null,
          history_original_seq: provenance.historyOriginalSeq ?? null,
          history_replacement_seq: provenance.historyReplacementSeq ?? null,
        })
        if (provenance && canonicalRequestJson(provenance.retained) !== canonicalRequestJson(block.content)) {
          missing.push(`${basePath}:retention-provenance-mismatch`)
        }
        if (provenance?.historyReplacementSeq !== undefined && block.content.length === 0) {
          stringLeaves(provenance.source, "/content", (leafPath, original) => {
            add(`${basePath}${leafPath}`, "", `${origin}${leafPath}`, original, removedParts)
          })
        }
        stringLeaves(block, "", (leafPath, text) => {
          const original = provenance && leafPath.startsWith("/content/")
            ? atPath(provenance.source, leafPath.slice("/content".length)) : text
          if (typeof original !== "string") { missing.push(`${basePath}${leafPath}:source-text`); return }
          add(`${basePath}${leafPath}`, text, origin === null ? null : `${origin}${leafPath}`, original)
        })
      })
    }
    const serialized = canonicalRequestJson(snapshot)
    const requestHash = hash(serialized)
    const snapshotId = `${facts.logicalCallId}:request`
    // Retries keep the first request snapshot and its original event cutoff.
    const previous = state.snapshots.get(snapshotId)
    if (previous === undefined && facts.attemptOrdinal !== 1) missing.push("snapshot:initial-attempt-missing")
    if (previous && previous.request_hash !== requestHash) missing.push("snapshot:request-hash-mismatch")
    const record = {
      schema_version: 1, record_type: "request/content", time: this.now(), session_id: sessionId,
      logical_call_id: facts.logicalCallId, attempt_id: facts.attemptId, attempt_ordinal: facts.attemptOrdinal,
      snapshot_id: snapshotId, snapshot: previous === undefined,
      cutoff_native_seq: state.nativeSeq,
      snapshot_cutoff_native_seq: previous?.cutoff_native_seq ?? state.nativeSeq,
      capture_boundary: "dsh-llm-stream-entry", provider_wire_attempt: "unknown",
      unit: "utf8-bytes", normalization: "utf16-to-well-formed-v1",
      content_scope: "message-content-string-leaves+system+canonical-tool-schemas-v1",
      retention_policy: policies().has(sessionId)
        ? { kind: "tool-result-utf8-prefix-v1", max_utf8_bytes: policies().get(sessionId) }
        : { kind: "full-client-request" },
      history_keep_recent_tool_results: historyPolicies().get(sessionId) ?? null,
      status: missing.length === 0 ? "complete" : "missing", missing_fields: [...new Set(missing)],
      request_hash: requestHash, request_utf8_bytes: Buffer.byteLength(serialized, "utf8"), request: snapshot,
      parts, removed_parts: removedParts, retention,
    }
    this.writeRecord(sessionId, record)
    if (previous === undefined) state.snapshots.set(snapshotId, { request_hash: requestHash, cutoff_native_seq: state.nativeSeq })
    return record
  }
}
